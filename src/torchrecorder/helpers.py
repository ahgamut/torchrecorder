# -*- coding: utf-8 -*-
"""
    torchrecorder.helpers
    ~~~~~~~~~~~~~~~~

    Helper functions for torchrecorder

    :copyright: (c) 2020 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from torch import randn
from .recorder import Recorder
from .renderer.gv import GraphvizRenderer
from graphviz import Digraph


def render_network(
    net,
    name,
    input_shapes,
    directory,
    filename=None,
    fmt="svg",
    input_data=None,
    render_depth=1,
    **styler_args
):
    """Render the structure of a `torch.nn.Module` to an image via `graphviz`.

    Args:
        net (`torch.nn.Module`):
        name (str): name of the network
        input_shapes (None, tuple or list(tuple)):
                    `tuple` if ``net`` has a single input,
                    `list` ( `tuple` ), `None`
                    if ``input_data`` is provided
        directory (str): directory to store the rendered image
        fmt (str, optional): image format
        input_data (`torch.Tensor` or `tuple` (`torch.Tensor` ), optional):
                    if ``net`` requires normalized inputs,
                    provide them here instead of setting ``input_shapes``.
        render_depth (int, optional): Default ``1``.
        **styler_args : node attributes to pass to `graphviz`

    """
    net = net.cpu().train()
    rec = record(net, name, input_shapes, input_data)
    g = make_dot(rec, render_depth, styler_cls=None, **styler_args)
    g.format = fmt
    g.attr(label="{} at depth = {}".format(name, render_depth))
    g.render("{}-{}".format(name, render_depth), directory=directory, cleanup=True)


def record(net, name, input_shapes, input_data=None):
    """Record the graph by running a single pass of a `torch.nn.Module`.

    Args:
        net (`torch.nn.Module`):
        name (str): name of the network
        input_shapes (None, tuple or list(tuple)):
                    `tuple` if ``net`` has a single input,
                    `list` ( `tuple` ), `None`
                    if ``input_data`` is provided
        input_data (`torch.Tensor` or `tuple` (`torch.Tensor` ), optional):
                    if ``net`` requires normalized inputs,
                    provide them here instead of setting ``input_shapes``.

    Returns:
        a `~.Recorder` object containing the execution graph

    """
    rec = Recorder()
    rec.register_hooks(net, depth=0, parent=None, name=name)

    data = []
    data_given = input_data is not None
    single_input = False
    if data_given:
        data = input_data
        single_input = not isinstance(input_data, tuple)
    else:
        if isinstance(input_shapes, list):
            single_input = False
            for shape in input_shapes:
                d = randn(shape)
                data.append(d)
            data = tuple(data)
        elif isinstance(input_shapes, tuple):
            single_input = True
            data = randn(input_shapes)

    if single_input:
        data.requires_grad = True
        rec.add_node(data, depth=0, parent=None, name="Input")
        pred = net(data)
    else:
        for i, d in enumerate(data):
            d.requires_grad = True
            rec.add_node(d, depth=0, parent=None, name="Input-{i}".format(i=i + 1))
        pred = net(*data)

    single_output = not isinstance(pred, tuple)
    if single_output:
        rec.nodes[pred].name = "Output"
    else:
        for i, p in enumerate(pred):
            rec.nodes[p].name = "Output-{i}".format(i=i + 1)

    rec.remove_hooks()
    return rec


def make_dot(rec, render_depth=256, styler_cls=None, **styler_args):
    """ Produces Graphviz representation from a `~torchrecorder.recorder.Recorder` object

    Args:
        rec (`~torchrecorder.recorder.Recorder`\ ):
        render_depth (int):     depth until which nodes should be rendered
        styler_cls:             styler class to instantiate when styling nodes.
                                If `None`, defaults to `.GraphvizStyler`.
    Kwargs:
        styler_args (optional): styler properties to be set for all nodes

    Returns:
        a `graphviz.Digraph` with the rendered nodes

    """
    graph_attr = dict(compound="true", ranksep="0.5", fontsize="24")
    node_attr = dict(fontsize="20")
    if styler_args.get("fontname", None) is not None:
        graph_attr["fontname"] = styler_args.get("fontname")
        node_attr["fontname"] = styler_args.get("fontname")
    g = Digraph(graph_attr=graph_attr, node_attr=node_attr)
    renderer = GraphvizRenderer(
        rec=rec, render_depth=render_depth, styler_cls=styler_cls, **styler_args
    )
    return renderer(g)
