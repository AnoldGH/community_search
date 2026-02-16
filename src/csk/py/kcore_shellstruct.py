from pathlib import Path

import click
import networkit as nk
import numba
import numpy as np
import pandas as pd
import scipy.sparse as sp

"""
K-core with an index structure based on ShellStruct (N. Barbieri, F. Bonchi, E. Galimberti, and F. Gullo. Efficient and effective community search. DMKD, 29(5):1406–1433, 2015.)
Implement the advanced algorithm to build the index (Y. Fang, R. Cheng, S. Luo, and J. Hu. Effective community search for large attributed graphs. PVLDB, 9(12):1233–1244, Aug. 2016.)
"""


# Disjoint set which stores anchor nodes
class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}  # each node starts as its own parent

    def find(self, item):
        if self.parent[item] == item:
            return item
        self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)
        # merge if the two roots are different. TODO: optimize for balanced tree
        if root1 != root2:
            self.parent[root1] = root2


class CLTreeNode:
    def __init__(self, core_num, vertices):
        self.core_num = core_num
        self.vertex_set = set(vertices)
        self.children = set()

    def add_child(self, child):
        self.children.add(child)


class AdvancedIndexBuilder:
    def __init__(self, graph):
        self.graph = graph
        self.v_to_treenode = dict()  # node_id -> CL-tree node
        self.anchor_map = (
            dict()
        )  # root -> anchor node_id (i.e., the node with least core number in the set)

    def build(self):
        core = nk.centrality.CoreDecomposition(self.graph).run()
        k_max = core.maxCoreNumber()
        k_shells = core.getPartition()

        top_level_treenodes = set()

        # initialize union-find and anchor map
        nodelist = set()
        for v in self.graph.iterNodes():
            self.anchor_map[v] = v
            nodelist.add(v)
        uf = UnionFind(nodelist)

        # bottom-up: decrement k in each iteration
        for k in range(k_max, -1, -1):
            # (new) nodes encountered for this k
            k_shell_nodes = k_shells.getMembers(
                k
            )  # set of nodes with core number exactly being k
            # print(f"{k}-shell: {k_shell_nodes}")

            if not k_shell_nodes:
                continue  # no nodes for this layer

            # find connected components at this level, taking previous CCs into consideration
            k_prime_nodes = set()  # all connected components in k'-cores where k' < k
            for v in k_shell_nodes:
                k_prime_nodes.add(uf.find(v))

            k_union_nodes = (
                k_shell_nodes | k_prime_nodes
            )  # nodes used to compute k-core
            k_union_subgraph = nk.graphtools.subgraphFromNodes(
                self.graph, k_union_nodes
            )
            k_union_cc = (
                nk.components.ConnectedComponents(k_union_subgraph)
                .run()
                .getComponents()
            )

            # print(f"{k}_union_cc: {k_union_cc}")
            for cc_nodes in k_union_cc:
                cc_nodes = set(cc_nodes)
                cc_union_nodes = cc_nodes & k_shell_nodes

                if len(cc_union_nodes) == 0:
                    continue  # empty connected components

                k_cc_treenode = CLTreeNode(
                    k, cc_union_nodes
                )  # CL-tree node for this k-shell component
                top_level_treenodes.add(k_cc_treenode)

                # to prevent repetitively adding the same CL-tree node
                processed_treenodes = set()

                for v in cc_union_nodes:
                    self.v_to_treenode[v] = k_cc_treenode

                    for u in self.graph.iterNeighbors(v):

                        # update CL-tree
                        if core.score(u) > core.score(v):
                            u_root = uf.find(u)
                            u_anchor = self.anchor_map[u_root]
                            prev_treenode = self.v_to_treenode[u_anchor]

                            if (
                                prev_treenode in processed_treenodes
                                or prev_treenode is k_cc_treenode
                            ):
                                continue  # skip repetitive and identical treenode

                            k_cc_treenode.add_child(prev_treenode)
                            top_level_treenodes.remove(prev_treenode)
                            processed_treenodes.add(prev_treenode)

                            # print(
                            #     f"Adding {prev_treenode} ({prev_treenode.vertex_set}) as a child to {k_cc_treenode} ({k_cc_treenode.vertex_set})"
                            # )

                        # update union-find structure
                        if core.score(u) >= core.score(v):
                            uf.union(u, v)

                    # update anchor if v has a strictly smaller core number
                    v_root = uf.find(v)
                    if core.score(self.anchor_map[v_root]) > core.score(v):
                        self.anchor_map[v_root] = v

        # TODO: build root node
        root = CLTreeNode(-1, [])
        for treenode in top_level_treenodes:
            root.add_child(treenode)

        # q = [root]
        # while len(q) != 0:
        #     node = q.pop(0)
        #     print(
        #         f"TreeNode: {node} (core={node.core_num}), with vertices: {node.vertex_set}"
        #     )
        #     q.extend(node.children)

        return None


@click.group()
def kcore():
    pass


@kcore.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def index(edgelist, output):
    graph = nk.graphio.EdgeListReader(",", 0).read(edgelist)
    core = nk.centrality.CoreDecomposition(graph).run()

    # obtain index
    indexer = AdvancedIndexBuilder(graph)
    indexer.build()

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
            q, k = line.strip().split(" ")
            outpath = Path(outputdir) / f"{q}/kcore_k{k}.txt"
            outpath.parent.mkdir(parents=True, exist_ok=True)
            with outpath.open("w") as outfile:
                print_kcore(int(q), int(k), outfile)


if __name__ == "__main__":
    kcore()
