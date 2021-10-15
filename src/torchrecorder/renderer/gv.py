# -*- coding: utf-8 -*-
"""
    torchrecorder.renderer.gv
    ~~~~~~~~~~~~~~~~~~~~

    Graphviz renderer object

    :param copyright: (c) 2020 by Gautham Venkatasubramanian.
    :param license: see LICENSE for more details.
"""
from ..nodes import BaseNode, TensorNode, ParamNode, OpNode, LayerNode
from .base import BaseRenderer
from graphviz import Digraph


class GraphvizStyler(object):
    """Provide styling options before rendering to graphviz.

    Attributes:
        styles (dict):   contains style properties for each subclass of `~torchrecorder.nodes.BaseNode`
    """

    def __init__(self, **styler_args):
        def setdefault(x, y):
            if styler_args.get(x, None) is None:
                styler_args[x] = y

        setdefault("style", "filled")
        setdefault("align", "left")
        setdefault("color", "black")

        self.styles = {
            BaseNode: dict(**styler_args),
            TensorNode: dict(**styler_args, fillcolor="lightblue"),
            ParamNode: dict(**styler_args, fillcolor="darkolivegreen"),
            OpNode: dict(**styler_args, fillcolor="orange", shape="box"),
            LayerNode: dict(**styler_args, fillcolor="lightgrey", shape="box"),
        }

    def style_node(self, node):
        """Construct style properties for the given node.

        Can be overridden to perform custom styling.

        Args:
            node (`~torchrecorder.nodes.BaseNode`\ ):
        Returns:
            a `dict` containing the required style properties

        """
        z = dict(**self.styles[type(node)])
        if isinstance(node, TensorNode):
            z["label"] = node.name + "\n" + str(list(node.fn.shape))
        else:
            z["label"] = node.name
        return z

    def style_edge(self, fnode, tnode):
        """Construct style properties to render the given edge

        Args:
            fnode: `~torchrecorder.nodes.BaseNode`
            tnode: `~torchrecorder.nodes.BaseNode`

        Returns:
            a `dict` containing the required style properties

        """
        return {}


class GraphvizRenderer(BaseRenderer):
    """Render information from a `~torchrecorder.recorder.Recorder` into a `graphviz.Digraph`.

    Attributes:
        styler (`class`): `.GraphvizStyler` or a subclass
    """

    def __init__(self, rec, render_depth=256, styler_cls=None, **styler_args):
        BaseRenderer.__init__(self, rec, render_depth)
        if styler_cls is None:
            styler_cls = GraphvizStyler
        self.styler = styler_cls(**styler_args)
        self.recursion_trace = []

    def render_node(self, g, node):
        """Render a node in `graphviz`

        Renders ``node`` into the `~graphviz.Digraph` ``g``,
        after applying appropriate styling.
        If ``node`` is a `~torchrecorder.nodes.LayerNode`, checks
        `.render_depth` to see if its
        `~.torchrecorder.nodes.LayerNode.subnets` have to rendered.

        Args:
            g (`graphviz.Digraph`):
            node (`~torchrecorder.nodes.BaseNode`):

        """
        if isinstance(node, LayerNode) and node.depth < self.render_depth:
            self.recursion_trace.append(g)
            self.render_recursive_node(g, node)
            self.recursion_trace.remove(g)
        else:
            style = self.styler.style_node(node)
            g.node(name=str(id(node)), **style)

    def render_recursive_node(self, g, node):
        """Render a `~torchrecorder.nodes.LayerNode` and its subnets.

        Args:
            g (`graphviz.Digraph`):
            node (`~torchrecorder.nodes.LayerNode`): has a
                                `~torchrecorder.nodes.LayerNode.depth` greater than
                                `.render_depth`

        The ``node`` is rendered as a separate `~graphviz.Digraph`
        and then is added as a `graphviz.Digraph.subgraph` to ``g``.
        """
        subg_style = self.styler.style_node(node)
        subg_style["fillcolor"] = "white"
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
        """Render an edge in `graphviz`

        Args:
            g (`graphviz.Digraph`):
            fnode (`~torchrecorder.nodes.BaseNode`):
            tnode (`~torchrecorder.nodes.BaseNode`):

        """
        style = self.styler.style_edge(fnode, tnode)
        g.edge(str(id(fnode)), str(id(tnode)), **style)
