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

    def __getstate__(self):
        """Iteratively flatten the CL-tree for pickle serialization,
        avoiding deep recursion that causes SIGSEGV on large graphs."""
        # Assign each CLTreeNode a unique integer id via BFS
        node_to_id = {}
        queue = []
        # Collect all tree nodes reachable from v_to_treenode values
        for treenode in self.v_to_treenode.values():
            if id(treenode) not in node_to_id:
                # Walk up to the root first
                root = treenode
                while root.parent is not None:
                    root = root.parent
                # BFS from root
                bfs = [root]
                while bfs:
                    cur = bfs.pop(0)
                    if id(cur) not in node_to_id:
                        node_to_id[id(cur)] = len(node_to_id)
                        bfs.extend(cur.children)

        id_to_node = {v: k for k, v in node_to_id.items()}
        # Build a flat list: each entry is (core_num, vertex_set, parent_id, [child_ids])
        obj_map = {}  # python id(obj) -> obj, for lookup
        for treenode in self.v_to_treenode.values():
            obj_map[id(treenode)] = treenode
            if treenode.parent is not None:
                obj_map[id(treenode.parent)] = treenode.parent
        # BFS again to capture all nodes including those not in v_to_treenode
        for py_id in list(node_to_id.keys()):
            if py_id in obj_map:
                for child in obj_map[py_id].children:
                    obj_map[id(child)] = child

        flat_nodes = [None] * len(node_to_id)
        for py_id, idx in node_to_id.items():
            node = obj_map[py_id]
            parent_idx = node_to_id[id(node.parent)] if node.parent is not None else -1
            child_idxs = [node_to_id[id(c)] for c in node.children]
            flat_nodes[idx] = (node.core_num, node.vertex_set, parent_idx, child_idxs)

        # Map v_to_treenode: vertex -> flat index
        v_to_idx = {v: node_to_id[id(tn)] for v, tn in self.v_to_treenode.items()}

        return {"flat_nodes": flat_nodes, "v_to_idx": v_to_idx}

    def __setstate__(self, state):
        """Reconstruct the CL-tree iteratively from the flat representation."""
        flat_nodes = state["flat_nodes"]
        v_to_idx = state["v_to_idx"]

        # Rebuild CLTreeNode objects without parent/child links first
        tree_nodes = []
        for core_num, vertex_set, _, _ in flat_nodes:
            tree_nodes.append(CLTreeNode(core_num, vertex_set))

        # Wire up parent/child pointers
        for idx, (_, _, parent_idx, child_idxs) in enumerate(flat_nodes):
            node = tree_nodes[idx]
            if parent_idx != -1:
                node.parent = tree_nodes[parent_idx]
            for ci in child_idxs:
                node.children.add(tree_nodes[ci])

        # Rebuild v_to_treenode
        self.v_to_treenode = {v: tree_nodes[idx] for v, idx in v_to_idx.items()}
        self.graph = None
        self.anchor_map = {}

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
    reader = nk.graphio.EdgeListReader("\t", 0, continuous=False)
    graph = reader.read(edgelist)
    # getNodeMap: original_id (str) -> internal_id (int)
    node_map = reader.getNodeMap()
    # Build reverse map: internal_id -> original_id (as int)
    reverse_map = {v: int(k) for k, v in node_map.items()}
    print(f"Graph loaded: {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")

    # obtain index
    indexer = AdvancedIndexBuilder(graph)
    indexer.build()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump({"indexer": indexer, "reverse_map": reverse_map, "node_map": node_map},
                     f, protocol=pickle.HIGHEST_PROTOCOL)


@kcore.command()
@click.option("--index", required=True, type=click.Path(exists=True))
@click.option("--nodelist", required=True, type=click.Path(exists=True))
@click.option("--outputdir", required=True, type=click.Path())
def search(index, nodelist, outputdir):
    with open(index, "rb") as f:
        data = pickle.load(f)
    # Support both old format (bare indexer) and new format (dict with maps)
    if isinstance(data, dict):
        indexer = data["indexer"]
        reverse_map = data.get("reverse_map")  # internal_id -> original_id
        node_map = data.get("node_map")        # original_id (str) -> internal_id
    else:
        indexer = data
        reverse_map = None
        node_map = None
    print(f"Indexer loaded")

    def ancestors(node):
        """Return the set of all ancestors (including the node itself)."""
        result = set()
        while node is not None:
            result.add(node)
            node = node.parent
        print(f"Ancestor found for node {node}")
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

    count = 0
    with open(nodelist) as nodefile:
        for line in nodefile.readlines():
            parts = line.strip().split(" ")
            queries_str = parts[0]
            k = int(parts[1]) if len(parts) > 1 else -1
            original_queries = [int(q) for q in queries_str.split(",")]
            # Map original IDs to internal IDs if node_map exists
            if node_map is not None:
                queries = []
                for q in original_queries:
                    if str(q) in node_map:
                        queries.append(node_map[str(q)])
                    else:
                        print(f"Warning: query node {q} not found in graph")
                        queries.append(-1)  # will be caught by find_kcore
            else:
                queries = original_queries

            resolved_k, component = find_kcore(queries, k)
            # Map internal IDs back to original IDs in output
            if reverse_map is not None:
                component = [reverse_map.get(v, v) for v in component]
            # Use query string as dir name, but fall back to a descriptive
            # name derived from the nodelist filename when it's too long for
            # the filesystem (max 255 bytes).
            dirname = queries_str
            if len(dirname.encode("utf-8")) > 255:
                dirname = Path(nodelist).stem + f"_line{count}"
            outpath = Path(outputdir) / f"{dirname}/kcore_k{resolved_k}.txt"
            outpath.parent.mkdir(parents=True, exist_ok=True)

            with outpath.open("w") as outfile:
                outfile.write("\n".join(map(str, component)))
                if len(component):
                    outfile.write("\n")
                #outfile.write("-1")

            count += 1
            if count % 50 == 0: 
                print(f"Finished {count} queries...")

    print("Search jobs completed.")

if __name__ == "__main__":
    kcore()
