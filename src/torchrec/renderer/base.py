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
    def __init__(self, rec, render_depth=256):
        self.rec = rec
        self.processed = OrderedDict()
        self.render_depth = render_depth

    def _render_node(self, dest, node):
        raise NotImplementedError("Base Class")

    def _render_edge(self, dest, fnode, tnode):
        raise NotImplementedError("Base Class")

    def _render_recursive_node(self, dest, node):
        raise NotImplementedError("Base Class")

    def __call__(self, dest):
        self._process_nodes()
        self._process_edges()
        while len(self.processed) != 0:
            node = next(iter(self.processed))
            targets = self.processed[node]
            self._render_node(dest, node)
            for t in targets:
                self._render_edge(dest, node, t)
            self.processed.pop(node)
        return dest

    def _process_nodes(self):
        for k, v in self.rec.nodes.items():
            if k is not None and v.depth <= self.render_depth:
                self.processed[v] = []
        self.processed = OrderedDict(
            sorted(
                self.processed.items(),
                key=lambda kv: kv[0].depth - int(isinstance(kv[0], LayerNode)),
            )
        )

    def _process_edges(self):
        for f, t in self.rec.edges:
            fnode = self.rec.nodes[f]
            tnode = self.rec.nodes[t]
            while fnode.depth > self.render_depth:
                fnode = self.rec.nodes[fnode.parent]
            while tnode.depth > self.render_depth:
                tnode = self.rec.nodes[tnode.parent]
            if fnode != tnode:
                self.processed[fnode].append(tnode)
