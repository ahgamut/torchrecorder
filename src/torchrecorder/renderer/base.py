# -*- coding: utf-8 -*-
"""
    torchrecorder.renderer.base
    ~~~~~~~~~~~~~~~~~~~~~~

    Abstract base class for Renderer objects

    :param copyright: (c) 2020 by Gautham Venkatasubramanian.
    :param license: see LICENSE for more details.
"""
from collections import OrderedDict
from ..nodes import LayerNode


class BaseRenderer(object):

    """Base Class for rendering information from a `~torchrecorder.recorder.Recorder`.

    Attributes:

        rec (`~torchrecorder.recorder.Recorder`):
        render_depth (int): nodes having a greater depth than this value
                            will not be rendered
        processed (`collections.OrderedDict`):
                            An ``OrderedDict`` whose keys contain ``nodes`` and values
                            contain the corresponding (directed) edge lists
    """

    def __init__(self, rec, render_depth=256):
        self.rec = rec
        self.render_depth = render_depth
        self.processed = OrderedDict()

    def render_node(self, dest, node):
        raise NotImplementedError("Base Class")

    def render_recursive_node(self, dest, node):
        raise NotImplementedError("Base Class")

    def render_edge(self, dest, fnode, tnode):
        raise NotImplementedError("Base Class")

    def __call__(self, dest):
        """Render nodes and edges.

        Uses `.render_depth` to select nodes and edges from `.rec`
        required for rendering. calls `.render_node` on each node,
        followed by `render_edge` on each edge from that node.

        Args:
            dest:    destination for the rendered information
                    (`graphviz.Digraph`, `dict` etc.)
        Returns:
            ``dest`` after updating with necessary information
        """
        self.processed.clear()
        self._process_nodes()
        self._process_edges()
        while len(self.processed) != 0:
            node = next(iter(self.processed))
            targets = self.processed[node]
            self.render_node(dest, node)
            for t in targets:
                self.render_edge(dest, node, t)
            self.processed.pop(node)
        return dest

    def _process_nodes(self):
        """Filter out nodes that have a greater depth than required.
        """
        for k, v in self.rec.nodes.items():
            if k is not None and v.depth <= self.render_depth:
                self.processed[v] = []

    def _process_edges(self):
        """Construct necessary edges between filtered nodes.

        After all nodes deeper than `.render_depth` have been removed, all the
        edges that between these nodes and those that remain must be
        transformed accordingly: such edges are "lifted up" for rendering, and
        ignored if they are internal to a node (i.e. both source and
        destination have been removed).
        """

        def lifted_node(x):
            xnode = self.rec.nodes[x]
            while xnode.depth > self.render_depth:
                xnode = self.rec.nodes[xnode.parent]
            return xnode

        lifted_edges = set(
            (lifted_node(x), lifted_node(y), z)
            for x, y, z in self.rec.edges
            if lifted_node(x) != lifted_node(y)
        )
        for fnode, tnode, _ in lifted_edges:
            self.processed[fnode].append(tnode)
