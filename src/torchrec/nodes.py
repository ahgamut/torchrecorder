# -*- coding: utf-8 -*-
"""
    torchrec.nodes
    ~~~~~~~~~~~~~~

    Nodes of the execution graph

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""


class BaseNode(object):
    def __init__(self, name="", fn=None, depth=-1, parent=None):
        self.name = name
        self.fn = fn
        self.depth = depth
        self.parent = parent

    def __str__(self):
        return " ".join([self.name, str(self.depth)])


class TensorNode(BaseNode):
    pass


class ParamNode(TensorNode):
    pass


class OpNode(BaseNode):
    pass


class LayerNode(BaseNode):
    def __init__(self, name="", fn=None, depth=-1, parent=None):
        BaseNode.__init__(self=self, name=name, fn=fn, depth=depth, parent=parent)
        self.pre = None
        self.post = None
        self.back = None
        self.subnets = set()
