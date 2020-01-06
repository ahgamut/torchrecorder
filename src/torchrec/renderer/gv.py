# -*- coding: utf-8 -*-
"""
    torchrec.renderer.gv
    ~~~~~~~~~~~~~~~~~~~~

    Graphviz renderer object

    :copyright: (c) 2020 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from torchrec.recorder import Recorder
from ..nodes import BaseNode, TensorNode, ParamNode, OpNode, LayerNode
from collections import OrderedDict
from graphviz import Digraph


class GraphvizStyler(object):
    def __init__(self, **default):
        self.styles = {}

        def setdefault(x, y):
            if default.get(x, None) is None:
                default[x] = y

        setdefault("style", "filled")
        setdefault("align", "left")
        setdefault("color", "black")

        self.styles[TensorNode] = dict(**default, fillcolor="lightblue")
        self.styles[ParamNode] = dict(**default, fillcolor="darkolivegreen")
        self.styles[OpNode] = dict(**default, fillcolor="orange", shape="box")
        self.styles[LayerNode] = dict(**default, fillcolor="lightgrey", shape="box")

    def __call__(self, node):
        z = dict(**self.styles[type(node)])
        if isinstance(node, TensorNode):
            z["label"] = node.name + "\n" + str(list(node.fn.shape))
        else:
            z["label"] = node.name
        return z

    def style_edge(self, edge):
        return {}


class GraphvizRenderer(object):
    def __init__(self, rec, render_depth=256, **styler_defaults):
        self.rec = rec
        self.processed = OrderedDict()
        self.render_depth = render_depth
        self.styler = GraphvizStyler(**styler_defaults)

    def __call__(self, g):
        self._process_nodes()
        self._process_edges()
        while len(self.processed) != 0:
            node = next(iter(self.processed))
            targets = self.processed[node]
            self._render_node(g, node)
            for t in targets:
                self._render_edge(g, node, t)
            self.processed.pop(node)
        return g

    def _render_node(self, g, node):
        if isinstance(node, LayerNode) and node.depth < self.render_depth:
            self._render_recursive_node(g, node)
        else:
            style = self.styler(node)
            g.node(name=str(id(node)), **style)

    def _render_edge(self, g, fnode, tnode):
        # style = self.styler.style_edge(fnode, tnode)
        g.edge(str(id(fnode)), str(id(tnode)))

    def _render_recursive_node(self, g, node):
        subg_style = self.styler(node)
        subg_style["fillcolor"] = "white"
        subg_style["label"] = node.name
        subg = Digraph(
            name="cluster_" + str(id(node)),
            graph_attr=subg_style,
            node_attr={"group": str(node.depth)},
        )
        for s in node.subnets:
            fnode = self.rec.nodes[s]
            self._render_node(subg, fnode)
        for s in node.subnets:
            fnode = self.rec.nodes[s]
            for tnode in self.processed[fnode]:
                if fnode.depth == tnode.depth:
                    self._render_edge(subg, fnode, tnode)
                else:
                    self._render_edge(g, fnode, tnode)
            self.processed.pop(fnode)
        g.subgraph(subg)

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
