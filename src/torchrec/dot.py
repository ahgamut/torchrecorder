from graphviz import Digraph
from .renderer import GraphvizRenderer
from .record import record


def make_dot(rec, render_depth=256, **kwargs):
    """ Produces Graphviz representation of a NN in PyTorch using the autograd graph

    :rec: a torchrec.Recorder object
    :render_depth: depth until which nodes in the graph should be rendered
    :**kwargs: properties to be set for all nodes in graphviz (fontname, etc)
    :returns: a Graphviz.Digraph with the rendered nodes

    """
    graph_attr = dict(compound="true", ranksep="0.5", fontsize="24")
    node_attr = dict(fontsize="20")
    if kwargs.get("fontname", None) is not None:
        graph_attr["fontname"] = kwargs.get("fontname")
        node_attr["fontname"] = kwargs.get("fontname")
    g = Digraph(graph_attr=graph_attr, node_attr=node_attr)
    renderer = GraphvizRenderer(rec=rec, render_depth=render_depth, **kwargs)
    return renderer(g)
