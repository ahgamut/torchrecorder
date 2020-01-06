# -*- coding: utf-8 -*-
"""
    torchrec.recorder
    ~~~~~~~~~~~~~~~~~

    Uses hooks to record the traversal of the execution graph

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from torch.nn import Module
from collections import OrderedDict
from .nodes import BaseNode, TensorNode, ParamNode, OpNode, LayerNode


class Recorder(object):

    """Record and store execution graph info"""

    def __init__(self):
        self.nodes = OrderedDict()
        self.node_types = dict()
        self.node_set = set()
        self.edges = set()

        self._create_dummy()

    def _create_dummy(self):
        self.node_set.add(None)
        self.nodes[None] = BaseNode(fn=None, depth=-1, parent=None, name="ContextDummy")

    def add_node(self, net, depth=0, parent=None, name=None):
        classname = net.__class__.__name__
        if self.node_types.get(classname):
            self.node_types[classname] += 1
        else:
            self.node_types[classname] = 1

        if name is None:
            objname = classname
            if self.node_types[classname] > 1:
                objname = objname + "-" + str(self.node_types[classname])
        else:
            objname = name

        if isinstance(net, Module):
            if depth > 0 and name is not None:
                objname = objname + "".join(["\n(", classname, ")"])
            x = LayerNode(name=objname, fn=net, parent=parent, depth=depth)
            x.pre = net.register_forward_pre_hook(generate_prehook(self, x))
            x.post = net.register_forward_hook(generate_posthook(self, x))
            # x.back = net.register_backward_hook(generate_backhook(self, x))
        elif "Tensor" in classname:
            x = TensorNode(name=objname, fn=net, parent=parent, depth=depth)
        elif "Parameter" in classname:
            x = ParamNode(name=objname, fn=net, parent=parent, depth=depth)
        elif hasattr(net, "next_functions"):
            x = OpNode(name=objname, fn=net, parent=parent, depth=depth)
        else:
            raise RuntimeError("Cannot create node for " + str(net))

        self.nodes[net] = x
        self.node_set.add(net)
        if x.parent is not None:
            pnode = self.nodes[x.parent]
            pnode.subnets.add(net)

    def add_edge(self, _from, _to):
        if _from is None or _to is None:
            raise AssertionError("Cannot draw edge involving" + str((_from, _to)))
        edge = (_from, _to)
        if edge not in self.edges:
            self.edges.add(edge)


def op_acc(gf, rec, node):
    if gf in rec.node_set:
        pass
    else:
        if hasattr(gf, "variable"):
            rec.nodes[gf] = rec.nodes[gf.variable]
            rec.node_set.add(gf)
        elif hasattr(gf, "next_functions"):
            rec.add_node(gf, node.depth + 1, node.fn)
            for x, y in gf.next_functions:
                if x is not None:
                    op_acc(x, rec, node)
                    rec.add_edge(_from=x, _to=gf)


def tensor_acc(tensor, rec, node):
    if tensor in rec.node_set:
        pass
    else:
        rec.add_node(tensor, depth=node.depth, parent=node.parent)


def param_acc(param, rec, node):
    if param in rec.node_set:
        pass
    else:
        rec.add_node(param, depth=node.depth + 1, parent=node.fn)


def leaf_dummy(tensor, rec, node):
    dummy = tensor + 0
    rec.node_set.add(dummy.grad_fn)
    rec.node_set.add(dummy)
    rec.nodes[dummy] = rec.nodes[tensor]
    rec.nodes[dummy.grad_fn] = rec.nodes[tensor]
    return dummy


def generate_prehook(rec, node):
    def prehook(module, inputs):
        for name, param in module.named_parameters(recurse=False):
            param_acc(param, rec, node)
            if name is not None and name != "":
                rec.nodes[param].name = name
        a = tuple(inputs)  # same input appearing multiple times?
        new_inputs = []
        for x in a:
            gf = x.grad_fn
            op_acc(gf, rec, rec.nodes[node.parent])
            if not x.is_leaf and x not in rec.node_set:
                x = x.detach()
                x.requires_grad = True
            tensor_acc(x, rec, node)
            x2 = leaf_dummy(x, rec, node)
            if gf is not None:
                rec.add_edge(gf, x2)
            new_inputs.append(x2)
        if not isinstance(inputs, tuple):
            return new_inputs[0]
        else:
            return tuple(new_inputs)

    return prehook


def generate_posthook(rec, node):
    def posthook(module, inputs, outputs):
        b = tuple(outputs)
        new_outputs = []
        for x in b:
            gf = x.grad_fn
            if not x.is_leaf and x not in rec.node_set:
                if "Select" in x.grad_fn.__class__.__name__:
                    x = x.grad_fn(x).detach()
                    gf = gf.next_functions[0][0]
                else:
                    x = x.detach()
                x.requires_grad = True
                tensor_acc(x, rec, node)
            elif x in rec.node_set:
                if node.fn == rec.nodes[x].parent:
                    rec.nodes[x].depth -= 1
                    rec.nodes[x].parent = node.parent
                    node.subnets.remove(rec.nodes[x].fn)
            op_acc(gf, rec, node)
            x2 = leaf_dummy(x, rec, node)
            if gf is not None:
                rec.add_edge(gf, x)
            new_outputs.append(x2)
        if not isinstance(outputs, tuple):
            return new_outputs[0]
        else:
            return tuple(new_outputs)

    return posthook


def generate_backhook(rec, node):
    def backhook(module, grad_inputs, grad_outputs):
        pass

    return backhook
