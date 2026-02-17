from pathlib import Path

import click
import networkit as nk
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp


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
    edges = np.loadtxt(edgelist, dtype=np.int64)
    n = np.max(edges) + 1
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(2 * len(edges), dtype=np.int8)
    graph = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()

    cores = pd.read_csv(index, sep="\t", header=None, names=["node", "core"])
    cores = cores.sort_values("node")["core"].to_numpy()

    @numba.njit
    def find_kcore(indptr, indices, cores, q, k):
        if cores[q] < k:
            return np.empty(0, dtype=np.int64)

        mask = cores >= k
        visited = np.zeros(n, dtype=np.uint8)
        stack, sidx = np.empty(n, dtype=np.int64), 0
        out, oidx = np.empty(n, dtype=np.int64), 0

        stack[0], sidx = q, sidx + 1
        visited[q] = 1

        while sidx > 0:
            v, sidx = stack[sidx - 1], sidx - 1
            out[oidx], oidx = v, oidx + 1

            for i in range(indptr[v], indptr[v + 1]):
                u = indices[i]
                if visited[u] == 0 and mask[u] != 0:
                    visited[u] = 1
                    stack[sidx], sidx = u, sidx + 1

        return out[:oidx]

    def print_kcore(q, k, outfile):
        component = find_kcore(graph.indptr, graph.indices, cores, q, k)
        outfile.write("\n".join(map(str, component)))
        if len(component):
            outfile.write("\n")
        outfile.write("-1")

    with open(nodelist) as nodefile:
        for line in nodefile.readlines():
            parts = line.strip().split(",")
            q = int(parts[0])
            if len(parts) == 1:
                k = cores[q]
            else:
                k = int(parts[1])

            outpath = Path(outputdir) / f"{q}/kcore_k{k}.txt"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            with outpath.open("w") as outfile:
                print_kcore(int(q), int(k), outfile)


if __name__ == "__main__":
    kcore()
