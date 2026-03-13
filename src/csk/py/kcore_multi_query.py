from pathlib import Path

import click
import networkit as nk
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import collections
from itertools import groupby
import time


"""Note: this method solves the Multi-Query k-Core problem, which does not allow custom k's - it only works to find the k-core community a query node is in for the largest k possible."""


@click.group()
def kcore():
    pass


@numba.njit
def _init_partitions(n):
    partitions = numba.typed.List()
    for u in range(n):
        s = set()
        s.add(np.int64(u))
        partitions.append(s)
    return partitions


@numba.njit
def _update_partition(partitions, labels, v_sub, indptr, indices):
    """Updates partition labels to reflect new connected components after adding E_sub.
    v_sub: int64 array of vertices incident to new edges.
    indptr, indices: CSR arrays for the accumulated k-shell edges.
    labels[u] == -1 means unassigned.
    Returns (pg_rows, pg_cols): partition-graph edges discovered during BFS."""

    # typed list init trick: append+clear to set the element type
    pg_rows = numba.typed.List()
    pg_rows.append(np.int64(0))
    pg_rows.clear()
    pg_cols = numba.typed.List()
    pg_cols.append(np.int64(0))
    pg_cols.clear()
    new_roots = numba.typed.List()
    new_roots.append(np.int64(0))
    new_roots.clear()

    for v in v_sub:
        if labels[v] == -1:
            labels[v] = v
            new_roots.append(v)
        root_v = labels[v]

        Q = numba.typed.List()
        Q.append(v)
        qi = 0
        U = set()
        U.add(v)

        while qi < len(Q):
            u = Q[qi]
            qi += 1

            if labels[u] == -1:
                labels[u] = root_v

                partitions[u].clear()
                partitions[root_v].add(u)

            for idx in range(indptr[u], indptr[u + 1]):
                v2 = indices[idx]
                if labels[v2] == -1:
                    Q.append(v2)
                elif labels[v2] not in U:
                    Q.append(labels[v2])
                    U.add(labels[v2])
                    pg_rows.append(root_v)
                    pg_cols.append(
                        labels[v2]
                    )  # build partition graph. Each node is a partition and edge indicates path between two partitions

    return pg_rows, pg_cols, new_roots


@numba.njit
def _expand_component(partitions, roots):
    """Collect all members from the given partition roots into a sorted array."""
    members = set()
    for i in range(len(roots)):
        root = roots[i]
        members.add(root)
        for member in partitions[root]:
            members.add(member)
    out = np.empty(len(members), dtype=np.int64)
    j = 0
    for m in members:
        out[j] = m
        j += 1
    out.sort()
    return out


@kcore.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def index(edgelist, output):
    graph = nk.graphio.EdgeListReader("\t", 0).read(edgelist)
    core = nk.centrality.CoreDecomposition(graph).run()

    data = [[node, int(core.score(node))] for node in range(graph.numberOfNodes())]
    df = pd.DataFrame(data, columns=["node", "core"])

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, header=False, sep="\t")


@kcore.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--index", required=True, type=click.Path(exists=True))
@click.option("--nodelist", required=True, type=click.Path(exists=True))
@click.option("--outputdir", required=True, type=click.Path())
@click.option("--delimiter", default="\t")
def search(edgelist, index, nodelist, outputdir, delimiter="\t"):
    # load edges
    edges = np.loadtxt(edgelist, dtype=np.int64, delimiter=delimiter)
    n = np.max(edges) + 1
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])

    # load index
    cores = pd.read_csv(index, sep=delimiter, header=None, names=["node", "core"])
    cores = cores.sort_values("node")["core"].to_numpy()

    # graph stats
    edgevals = np.minimum(cores[row], cores[col])  # core number by edge
    order = np.argsort(edgevals)
    edgevals = edgevals[order]
    row = row[order]
    col = col[order]  # order edges by minimal coreness of incident vertices
    counts = np.bincount(edgevals)  # count edges with coreness k
    indices = np.cumsum(counts)  # quickly find the CSR indices of edges with coreness k

    # initialize labels and partitions
    labels = np.full(n, -1, dtype=np.int64)
    partitions = _init_partitions(n)

    # process the queries, sort by k values, fill in empty cells with largest possible k
    query_df = pd.read_csv(
        nodelist, sep=" ", header=None, names=["q", "k"], dtype=pd.Int64Dtype
    )
    replacement_qs = cores[query_df["q"].values]
    query_df["k"] = query_df["k"].fillna(
        pd.Series(replacement_qs, index=query_df.index)
    )
    query_df.sort_values(by="k", ascending=False, inplace=True)

    # group queries by k value (already sorted descending)
    # ks[0] = max possible core number (i.e., |V|), then each unique k from the queries
    ks = [n, *query_df["k"].values]
    ks = [k for k, _ in groupby(ks)]  # collapse duplicate k's together

    # map each k value to the queries at that level
    queries_at_k = collections.defaultdict(list)
    for _, r in query_df.iterrows():
        queries_at_k[int(r["k"])].append(int(r["q"]))

    # accumulated partition-graph edges (COO format)
    # partition-graph: each node is a partition, and each edge indicates a path between two partitions. Essentially node contraction through a separate graph
    all_pg_rows = []
    all_pg_cols = []
    active_roots_set = set()  # maintained incrementally

    # main loop
    for i in range(1, len(ks)):
        num_queries = len(queries_at_k.get(ks[i], []))
        print(f"--- i={i}, k={ks[i]}, k_prev={ks[i-1]}, queries={num_queries} ---")

        # build CSR directly from sorted edge slices
        t0 = time.time()
        lo = indices[ks[i] - 1] if ks[i] > 0 else 0
        hi = indices[min(ks[i - 1], len(indices)) - 1] if ks[i - 1] > 0 else 0
        print(f"  lo: {lo}, hi: {hi}")

        r_slice = row[lo:hi]
        c_slice = col[lo:hi]
        if len(r_slice) > 0:
            # sort by coreness -> sort by source nodes
            order_sub = np.argsort(r_slice)
            sorted_row = r_slice[order_sub]
            sorted_col = c_slice[order_sub].astype(np.int64)

            # construct CSR
            e_counts = np.bincount(sorted_row, minlength=n)
            e_indptr = np.empty(n + 1, dtype=np.int64)
            e_indptr[0] = 0
            np.cumsum(e_counts, out=e_indptr[1:])
        else:
            sorted_col = np.empty(0, dtype=np.int64)
            e_indptr = np.zeros(n + 1, dtype=np.int64)
        t1 = time.time()
        print(f"  build_csr: {t1-t0:.4f}s ({len(sorted_col)} edges)")

        # extract v_sub: nodes incident to the edges in the new k-shell batch
        t2 = time.time()
        v_sub = np.where(np.diff(e_indptr) > 0)[0].astype(np.int64)
        t3 = time.time()
        print(f"  v_sub (CSR): {t3-t2:.4f}s ({len(v_sub)} nodes)")

        # update partition information
        if len(v_sub) > 0:
            t4 = time.time()
            pg_rows, pg_cols, new_roots = _update_partition(
                partitions,
                labels,
                v_sub,
                e_indptr,
                sorted_col,
            )
            t5 = time.time()
            print(f"  update_partition: {t5-t4:.4f}s ({len(pg_rows)} pg edges)")

            # accumulate partition-graph edges and roots
            all_pg_rows.extend(pg_rows)
            all_pg_cols.extend(pg_cols)
            active_roots_set.update(new_roots)

        # answer all queries at this k level
        t7 = time.time()
        current_queries = queries_at_k.get(ks[i], [])
        if current_queries:
            # build partition graph from accumulated COO edges
            if all_pg_rows:
                pg_r = np.array(all_pg_rows, dtype=np.int64)
                pg_c = np.array(all_pg_cols, dtype=np.int64)
                pg_data = np.ones(len(pg_r), dtype=np.int8)
                pg_graph = sp.coo_matrix((pg_data, (pg_r, pg_c)), shape=(n, n)).tocsr()
                n_components, comp_labels = connected_components(
                    pg_graph, directed=False
                )
            else:
                comp_labels = np.arange(n, dtype=np.int64)

            # map component -> list of roots (lazy expansion)
            comp_roots = collections.defaultdict(list)  # which CC the root belongs to
            for root in active_roots_set:
                comp_roots[comp_labels[root]].append(root)

            # expanded cache: component -> full member set
            comp_members = {}

            for q in current_queries:
                # find the component of the query's partition root
                q_root = labels[q]
                if q_root == -1:
                    # query node not yet in any partition — singleton
                    community_arr = np.array([q], dtype=np.int64)
                else:
                    q_comp = comp_labels[q_root]
                    if q_comp not in comp_members:
                        # expand on first access (JIT-compiled)
                        roots = comp_roots.get(q_comp, np.empty(0, dtype=np.int64))
                        comp_members[q_comp] = _expand_component(partitions, roots)
                    community_arr = comp_members[q_comp]

                outpath = Path(outputdir) / f"{q}/kcore_k{ks[i]}.txt"
                outpath.parent.mkdir(parents=True, exist_ok=True)
                with outpath.open("w") as outfile:
                    outfile.write("\n".join(map(str, community_arr)))
                    if len(community_arr):
                        outfile.write("\n")
                    outfile.write("-1")
        t8 = time.time()
        print(f"  get_community + write ({num_queries} queries): {t8-t7:.4f}s")


if __name__ == "__main__":
    kcore()
