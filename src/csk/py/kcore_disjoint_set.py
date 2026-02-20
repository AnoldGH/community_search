from pathlib import Path

import click
import networkit as nk
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp


@numba.njit
def _bfs_kcore(indptr, indices, cores, seeds, k, n):
    """Multi-source BFS from seed nodes in k-core subgraph."""
    visited = np.zeros(n, dtype=np.uint8)
    stack = np.empty(n, dtype=np.int64)
    sidx = 0
    out = np.empty(n, dtype=np.int64)
    oidx = 0

    for i in range(len(seeds)):
        s = seeds[i]
        if cores[s] >= k and visited[s] == 0:
            visited[s] = 1
            stack[sidx] = s
            sidx += 1

    while sidx > 0:
        sidx -= 1
        v = stack[sidx]
        out[oidx] = v
        oidx += 1

        for i in range(indptr[v], indptr[v + 1]):
            u = indices[i]
            if visited[u] == 0 and cores[u] >= k:
                visited[u] = 1
                stack[sidx] = u
                sidx += 1

    return out[:oidx]


def _ds_find(parent, x):
    """Find representative from a given node."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return int(x)


@click.group()
def kcore():
    pass


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
def search(edgelist, index, nodelist, outputdir):
    # Read-only CSR graph (enables numba JIT on BFS)
    edges = np.loadtxt(edgelist, dtype=np.int64)
    n = int(np.max(edges) + 1)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(2 * len(edges), dtype=np.int8)
    graph = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

    cores = pd.read_csv(index, sep="\t", header=None, names=["node", "core"])
    cores = cores.sort_values("node")["core"].to_numpy()

    # Disjoint set on numpy array for memoizing past merges
    parent = np.arange(n, dtype=np.int64)
    members = {}  # representative -> np.array of member node ids
    result_cache = {}  # (representative, k) -> set of result nodes

    def union_all(result_nodes):
        """Union all result nodes and update members. Returns new representative."""
        if len(result_nodes) == 0:
            return -1
        rep = _ds_find(parent, result_nodes[0])
        for i in range(1, len(result_nodes)):
            r = _ds_find(parent, result_nodes[i])
            if r != rep:
                if r in members:
                    del members[r]
                parent[r] = rep
        members[rep] = result_nodes
        return rep

    def find_kcore(q, k):
        if cores[q] < k:
            return set()

        r = _ds_find(parent, q)
        if (r, k) in result_cache:
            return result_cache[(r, k)]

        # Multi-source BFS: seed from all members of r's group
        seeds = members.get(r, np.array([q], dtype=np.int64))
        result = _bfs_kcore(graph.indptr, graph.indices, cores, seeds, k, n)

        if len(result) > 0:
            rep = union_all(result)
            result_set = set(result.tolist())
            result_cache[(rep, k)] = result_set
        else:
            result_set = set()

        return result_set

    def print_kcore(q, k, outfile):
        component = find_kcore(q, k)
        outfile.write("\n".join(map(str, component)))
        if len(component):
            outfile.write("\n")

    query_df = pd.read_csv(nodelist, sep=" ", header=None, names=["q", "k"])
    replacement_qs = cores[query_df["q"].values]
    query_df["k"] = query_df["k"].fillna(
        pd.Series(replacement_qs, index=query_df.index)
    )
    query_df.sort_values(by="k", ascending=False, inplace=True)

    for _, query in query_df.iterrows():
        q = int(query["q"])
        k = int(query["k"])

        if k == -1:
            k = int(cores[q])

        outpath = Path(outputdir) / f"{q}/kcore_k{k}.txt"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with outpath.open("w") as outfile:
            print_kcore(q, k, outfile)


if __name__ == "__main__":
    kcore()
