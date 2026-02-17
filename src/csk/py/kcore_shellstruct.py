import pickle
from pathlib import Path

import click
import networkit as nk

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
        self.parent = None

    def add_child(self, child):
        self.children.add(child)
        child.parent = self


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

    # obtain index
    indexer = AdvancedIndexBuilder(graph)
    indexer.build()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(indexer, f, protocol=pickle.HIGHEST_PROTOCOL)


@kcore.command()
@click.option("--index", required=True, type=click.Path(exists=True))
@click.option("--nodelist", required=True, type=click.Path(exists=True))
@click.option("--outputdir", required=True, type=click.Path())
def search(index, nodelist, outputdir):
    with open(index, "rb") as f:
        indexer = pickle.load(f)

    def ancestors(node):
        """Return the set of all ancestors (including the node itself)."""
        result = set()
        while node is not None:
            result.add(node)
            node = node.parent
        return result

    def find_kcore(queries, k):
        """Returns (resolved_k, sorted_vertices). resolved_k is the actual
        core number used (matters when k=-1)."""
        nodes = []
        for q in queries:
            if q not in indexer.v_to_treenode:
                return k, []
            node = indexer.v_to_treenode[q]
            if k == -1:
                # just stay at the node - this is the highest k possible
                pass
            elif node.core_num < k:
                return k, []
            else:
                # walk up to the first ancestor with core_num <= k
                while node.parent and node.parent.core_num >= k:
                    node = node.parent
            nodes.append(node)

        # intersecting ancestor sets to find LCA
        lca_candidates = ancestors(nodes[0])
        for node in nodes[1:]:
            lca_candidates &= ancestors(node)

        if not lca_candidates:
            return k, []

        # LCA is the deepest common ancestor (highest core_num)
        lca = max(lca_candidates, key=lambda n: n.core_num)

        if k != -1 and lca.core_num < k:
            return k, []

        # collect all vertices from the LCA and its descendants
        vertices = set()
        stack = [lca]
        while stack:
            cur = stack.pop()
            vertices.update(cur.vertex_set)
            stack.extend(cur.children)
        return lca.core_num, sorted(vertices)

    with open(nodelist) as nodefile:
        for line in nodefile.readlines():
            parts = line.strip().split(" ")
            queries_str = parts[0]
            k = int(parts[1]) if len(parts) > 1 else -1
            queries = [int(q) for q in queries_str.split(",")]

            resolved_k, component = find_kcore(queries, k)
            outpath = Path(outputdir) / f"{queries_str}/kcore_k{resolved_k}.txt"
            outpath.parent.mkdir(parents=True, exist_ok=True)

            with outpath.open("w") as outfile:
                outfile.write("\n".join(map(str, component)))
                if len(component):
                    outfile.write("\n")
                outfile.write("-1")


if __name__ == "__main__":
    kcore()
