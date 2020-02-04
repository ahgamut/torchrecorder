# -*- coding: utf-8 -*-
"""
    torchrec.renderer.gv
    ~~~~~~~~~~~~~~~~~~~~

    Graphviz renderer object

    :copyright: (c) 2020 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from ..nodes import BaseNode, TensorNode, ParamNode, OpNode, LayerNode
from .base import BaseRenderer
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


class GraphvizRenderer(BaseRenderer):
    def __init__(
        self, rec, render_depth=256, styler_cls=GraphvizStyler, **styler_defaults
    ):
        BaseRenderer.__init__(self, rec, render_depth)
        self.styler = styler_cls(**styler_defaults)
        self.recursion_trace = []

    def render_node(self, g, node):
        if isinstance(node, LayerNode) and node.depth < self.render_depth:
            self.recursion_trace.append(g)
            self.render_recursive_node(g, node)
            self.recursion_trace.remove(g)
        else:
            style = self.styler(node)
            g.node(name=str(id(node)), **style)

    def render_recursive_node(self, g, node):
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
            self.render_node(subg, fnode)
        for s in node.subnets:
            fnode = self.rec.nodes[s]
            for tnode in self.processed[fnode]:
                if fnode.depth == tnode.depth:
                    self.render_edge(subg, fnode, tnode)
                else:
                    depth_diff = abs(fnode.depth - tnode.depth)
                    self.render_edge(self.recursion_trace[-depth_diff], fnode, tnode)
            self.processed.pop(fnode)
        g.subgraph(subg)

    def render_edge(self, g, fnode, tnode):
        # style = self.styler.style_edge(fnode, tnode)
        g.edge(str(id(fnode)), str(id(tnode)))


def make_dot(rec, render_depth=256, styler_cls=GraphvizStyler, **styler_args):
    """ Produces Graphviz representation of a NN in PyTorch using the autograd graph

    :rec:           a torchrec.Recorder object
    :render_depth:  depth until which nodes in the graph should be rendered
    :styler_cls:    (subclass of) GraphvizStyler, to be instantiated
                    by the renderer
    :**styler_args:      default styler properties to be set for
                    all nodes in graphviz (fontname, etc)
    :returns:       a Graphviz.Digraph with the rendered nodes

    """
    graph_attr = dict(compound="true", ranksep="0.5", fontsize="24")
    node_attr = dict(fontsize="20")
    if kwargs.get("fontname", None) is not None:
        graph_attr["fontname"] = kwargs.get("fontname")
        node_attr["fontname"] = kwargs.get("fontname")
    g = Digraph(graph_attr=graph_attr, node_attr=node_attr)
    renderer = GraphvizRenderer(
        rec=rec, render_depth=render_depth, styler_cls=styler_cls, **kwargs
    )
    return renderer(g)
