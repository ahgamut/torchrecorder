# -*- coding: utf-8 -*-
"""
    torchrec.renderer.base
    ~~~~~~~~~~~~~~~~~~~~~~

    Abstract base class for Renderer objects

    :copyright: (c) 2020 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from collections import OrderedDict
from ..nodes import LayerNode


class BaseRenderer(object):

    """Base Class for rendering information from a `Recorder` object.

    Attributes:

    :rec:           A `Recorder` object
    :render_depth:  nodes having a greater depth than this value
                    will not be rendered
    :processed:     An `OrderedDict` whose keys contain `nodes` and values
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

        :dest:      destination for the rendered information (graphviz.Digraph,
                    JSON/dict etc.)
        :returns:   `dest` after updating with necessary
                    information
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

        :returns: `None`
        """
        for k, v in self.rec.nodes.items():
            if k is not None and v.depth <= self.render_depth:
                self.processed[v] = []

    def _process_edges(self):
        """Construct necessary edges between filtered nodes.

        After all nodes deeper than `render_depth` have been removed, all the
        edges that between these nodes and those that remain must be
        transformed accordingly: such edges are "raised up" for rendering, and
        ignored if they are internal to a node (i.e. both source and
        destination have been removed).

        :returns: `None`

        """
        for f, t in self.rec.edges:
            fnode = self.rec.nodes[f]
            tnode = self.rec.nodes[t]
            while fnode.depth > self.render_depth:
                fnode = self.rec.nodes[fnode.parent]
            while tnode.depth > self.render_depth:
                tnode = self.rec.nodes[tnode.parent]
            if fnode != tnode:
                self.processed[fnode].append(tnode)
