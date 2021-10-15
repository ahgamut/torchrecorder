# -*- coding: utf-8 -*-
"""
    torchrecorder.nodes
    ~~~~~~~~~~~~~~

    Nodes of the execution graph

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""


class BaseNode(object):
    """Wrapper object to encapsulate recorded information.

    Attributes:
        fn  (object):       an `object` recorded by the `~torchrecorder.recorder.Recorder`
        name (str):         name of the `.fn`
        depth (int):        `int`, scope depth of `.fn`
        parent (object):    a `.fn` in whose scope the current `.fn` exists
    """

    def __init__(self, name="", fn=None, depth=-1, parent=None):
        self.fn = fn
        self.name = name
        self.depth = depth
        self.parent = parent

    def __str__(self):
        internals = [
            "depth={}".format(self.depth),
            "fn_type={}".format(type(self.fn)),
            "parent_type={}".format(type(self.parent)),
        ]
        return self.name + "(" + ",".join(internals) + ")"

    def __repr__(self):
        return self.__str__()


class TensorNode(BaseNode):
    """Node to encapsulate a `torch.Tensor`.

    Attributes:
        fn( `torch.Tensor` ):
        name (str):         name of the `.fn`
        depth (int):        `int`, scope depth of `.fn`
        parent (object):    a `.fn` in whose scope the current `.fn` exists
    """

    pass


class ParamNode(TensorNode):
    """Node to encapsulate a `torch.nn.Parameter`.

    Attributes:
        fn( `torch.nn.Parameter` ):
        name (str):         name of the `.fn`
        depth (int):        `int`, scope depth of `.fn`
        parent (object):    a `~torch.nn.Module` whose `~torch.nn.Module.parameters` contains `.fn`
    """

    pass


class OpNode(BaseNode):
    """Node to encapsulate an Op, a ``grad_fn`` attribute
    of a `torch.Tensor`.

    Attributes:
        fn( `torch.Tensor` ):
        name (str):         name of the `.fn`
        depth (int):        `int`, scope depth of `.fn`
        parent (object):    a `~torch.nn.Module` in whose ``forward`` the current `OpNode.fn` was executed
    """

    pass


class LayerNode(BaseNode):
    """Node to encapsulate a `torch.nn.Module`.

    Attributes:
        fn( `torch.nn.Module` ):
        name (str):         name of the `.fn`
        depth (int):        `int`, scope depth of `.fn`
        parent (object):    a `~torch.nn.Module` in whose `~torch.nn.Module.forward`
                            `.fn` was called
        subnets (set):      a set `~torch.nn.Module` s or ``grad_fn`` s which are
                            called in `.fn` 's `~torch.nn.Module.forward`
        pre :     ``handle`` to the prehook on `.fn`
        post :    ``handle`` to the hook on `.fn`
    """

    def __init__(self, name="", fn=None, depth=-1, parent=None):
        BaseNode.__init__(self=self, name=name, fn=fn, depth=depth, parent=parent)
        self.pre = None
        self.post = None
        self.back = None
        self.subnets = set()

    def __str__(self):
        internals = [
            "depth={}".format(self.depth),
            "fn_type={}".format(type(self.fn)),
            "parent_type={}".format(type(self.parent)),
            "subnets_in_scope={}".format([type(x) for x in self.subnets]),
        ]
        return self.name + "(" + ",".join(internals) + ")"
