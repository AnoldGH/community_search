from pathlib import Path

import click
import networkit as nk
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp

from collections import deque


class DisjointSet:
    """Union-Find for tracking node contractions with path compression."""

    def __init__(self):
        self.parent = {}

    def find(self, node):
        """Find the representative of node's set, with path compression."""
        if node not in self.parent:
            return node
        root = node
        while root in self.parent:
            root = self.parent[root]
        # path compression
        while node != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, from_node, to_node):
        """Make from_node resolve to to_node's representative."""
        self.parent[from_node] = self.find(to_node)


def expand_node(contracted_nodes, node):
    """Return the set of original nodes represented by a (possibly contracted) node."""
    if node in contracted_nodes:
        return set(contracted_nodes[node])
    return {node}


# networKit does not have a native "contract node" operation. Here is a custom one which may not be the most efficient.
# return the contracted node
def contract_node(graph, contracted_nodes, node_map, u, v):
    if not graph.hasNode(u) or not graph.hasNode(v):
        return

    # contract the node with lower degree to the node with higher degree to save time
    if graph.degree(u) > graph.degree(v):
        from_node = v
        to_node = u
    else:
        from_node = u
        to_node = v

    # migrate edges from one node to another
    for neighbor in graph.iterNeighbors(from_node):
        graph.addEdge(to_node, neighbor, checkMultiEdge=True)

    # update contracted_nodes
    if to_node not in contracted_nodes:
        contracted_nodes[to_node] = {to_node}

    if from_node in contracted_nodes:
        contracted_nodes[to_node].update(contracted_nodes[from_node])
        del contracted_nodes[from_node]
    else:
        contracted_nodes[to_node].add(from_node)

    # remove from node from the network
    node_map.union(from_node, to_node)
    graph.removeNode(from_node)

    return to_node


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
    graph = nk.graphio.EdgeListReader("\t", 0).read(edgelist)

    cores = pd.read_csv(index, sep="\t", header=None, names=["node", "core"])
    cores = cores.sort_values("node")["core"].to_numpy()

    contracted_nodes = dict()
    node_map = (
        DisjointSet()
    )  # map original node_id to the new node_id if it's contracted

    def find_kcore(q, k):
        if k < cores[q]:
            return set()

        q = node_map.find(q)

        queue = deque()
        queue.append(q)

        visited = set()
        result = expand_node(contracted_nodes, q)

        while len(queue) != 0:
            q = queue.popleft()
            if cores[q] < k:
                continue  # drop node not in k-core

            neighbors = list()
            for neighbor in graph.iterNeighbors(q):
                if neighbor == q:
                    continue  # ignore self-loops

                neighbors.append(neighbor)

            for neighbor in neighbors:
                if cores[neighbor] < k or neighbor in visited:
                    continue  # neighbor not in any k-core

                visited.add(neighbor)

                result.update(expand_node(contracted_nodes, neighbor))

                for next_neighbor in graph.iterNeighbors(neighbor):
                    queue.append(next_neighbor)

                q = contract_node(graph, contracted_nodes, node_map, q, neighbor)

        return result

    def print_kcore(q, k, outfile):
        component = find_kcore(q, k)
        outfile.write("\n".join(map(str, component)))
        if len(component):
            outfile.write("\n")
        # outfile.write("-1")

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

        print(f"Query: q={q}, k={k}")
        outpath = Path(outputdir) / f"{q}/kcore_k{k}.txt"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with outpath.open("w") as outfile:
            print_kcore(q, k, outfile)

        print(f"Current contracted_nodes count: {contracted_nodes}")


if __name__ == "__main__":
    kcore()
