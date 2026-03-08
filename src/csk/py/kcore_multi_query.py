from pathlib import Path

import click
import networkit as nk
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp
import collections
from itertools import groupby


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
def _update_partition(
    partitions, labels, new_edges, indptr, indices, cores, k_threshold
):
    """Updates partition labels to reflect new connected components after adding E_sub.
    new_edges: tuples of two int64 elements (v1, v2).
    indptr, indices: CSR arrays for neighbor lookup.
    cores: int64 array of core numbers per node.
    k_threshold: only follow edge (u, v2) if min(cores[u], cores[v2]) >= k_threshold.
    labels[u] == -1 means unassigned."""

    v_sub = set()
    for i in range(len(new_edges)):
        v_sub.add(new_edges[i, 0])
        v_sub.add(new_edges[i, 1])

    for v in v_sub:
        Q = numba.typed.List()
        Q.append(v)
        qi = 0
        U = set()
        U.add(v)

        while qi < len(Q):
            u = Q[qi]
            qi += 1

            if labels[u] == -1:
                labels[u] = v

                partitions[u].clear()
                partitions[v].add(u)

            for idx in range(indptr[u], indptr[u + 1]):
                v2 = indices[idx]
                if min(cores[u], cores[v2]) < k_threshold:
                    continue
                if labels[v2] == -1:
                    Q.append(v2)
                elif labels[v2] not in U:
                    Q.append(labels[v2])
                    U.add(labels[v2])


@numba.njit
def _get_community(partitions, labels, indptr, indices, cores, k_threshold, query):
    """Find the community containing query.
    indptr, indices: CSR arrays for neighbor lookup.
    cores: int64 array of core numbers per node.
    k_threshold: only follow edge (u, v2) if min(cores[u], cores[v2]) >= k_threshold.
    labels[u] == -1 means unassigned."""

    C = set()
    Q = numba.typed.List()
    Q.extend(partitions[labels[query]])
    qi = 0
    U = set()
    U.add(query)

    added_partition = set()

    while qi < len(Q):
        u = Q[qi]
        qi += 1

        # add a partition to returned community only on first encounter
        if labels[u] not in added_partition:
            added_partition.add(labels[u])

            for member in partitions[labels[u]]:
                C.add(member)

        for idx in range(indptr[u], indptr[u + 1]):
            v2 = indices[idx]
            if min(cores[u], cores[v2]) < k_threshold:
                continue
            if labels[v2] != -1 and labels[v2] not in U:
                Q.extend(partitions[labels[v2]])
                U.add(labels[v2])

    return C


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
    edges = np.loadtxt(edgelist, dtype=np.int64, delimiter=delimiter)
    n = np.max(edges) + 1
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(2 * len(edges), dtype=np.int8)
    graph = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

    # load index
    cores = pd.read_csv(index, sep=delimiter, header=None, names=["node", "core"])
    cores = cores.sort_values("node")["core"].to_numpy()

    # initialize labels and partitions
    labels = np.full(n, -1, dtype=np.int64)
    partitions = _init_partitions(n)

    # find k shell edges & incident nodes
    k_shell_edges = collections.defaultdict(set)
    for u, v in edges:
        target_k = min(cores[u], cores[v])
        k_shell_edges[target_k].add((u, v))

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
    for _, row in query_df.iterrows():
        queries_at_k[int(row["k"])].append(int(row["q"]))

    # main loop
    for i in range(1, len(ks)):
        added_edges = set()
        for j in range(ks[i], ks[i - 1]):
            added_edges = added_edges.union(k_shell_edges[j])

        if len(added_edges) > 0:
            added_edges_arr = np.array(list(added_edges), dtype=np.int64)
            _update_partition(
                partitions,
                labels,
                added_edges_arr,
                graph.indptr,
                graph.indices,
                cores,
                np.int64(ks[i]),
            )

        # answer all queries at this k level
        for q in queries_at_k.get(ks[i], []):
            community = _get_community(
                partitions,
                labels,
                graph.indptr,
                graph.indices,
                cores,
                np.int64(ks[i]),
                np.int64(q),
            )
            outpath = Path(outputdir) / f"{q}/kcore_k{ks[i]}.txt"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            with outpath.open("w") as outfile:
                outfile.write("\n".join(map(str, sorted(community))))
                if len(community):
                    outfile.write("\n")
                outfile.write("-1")


if __name__ == "__main__":
    kcore()
