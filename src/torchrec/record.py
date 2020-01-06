from torch import randn
from .recorder import Recorder


def register_hooks(net, rec, depth=0, parent=None, name=None):
    """Register the hooks of the recorder recursively on a torch.nn.Module

    :net: a torch.nn.Module
    :rec: a Recorder object
    :depth: int
    :parent: the parent Module of net
    :returns: None

    """
    rec.add_node(net, depth, parent, name)
    for n, x in net.named_children():
        register_hooks(x, rec, depth=depth + 1, parent=net, name=n)


def record(net, input_shapes, name):
    """Runs a single pass of of a torch.nn.Module and record execution

    :net: a torch.nn.Module
    :input_shapes: List(Tuple) if multiple inputs, else Tuple
    :name: name of the network
    :returns: a torchrec.Recorder object containing the execution graph

    """
    rec = Recorder()
    register_hooks(net, rec, depth=0, parent=None, name=name)

    data = []
    if isinstance(input_shapes, list):
        for i, shape in enumerate(input_shapes):
            d = randn(shape, requires_grad=True)
            rec.add_node(d, depth=0, parent=None, name="Input-{i}".format(i=i + 1))
            data.append(d)
        data = tuple(data)
        pred = net(*data)
    elif isinstance(input_shapes, tuple):  # singleton
        data = randn(input_shapes, requires_grad=True)
        rec.add_node(data, depth=0, parent=None, name="Input")
        pred = net(data)

    if isinstance(pred, tuple):
        for i, p in enumerate(pred):
            rec.nodes[p].name = "Output-{i}".format(i=i + 1)
    else:
        rec.nodes[pred].name = "Output"
    return rec
