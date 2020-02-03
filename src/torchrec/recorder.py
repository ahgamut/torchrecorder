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

    """Record and store execution graph info

    A `Recorder` object stores:
        * a set of objects (`fn`s) that contain recordable information
        * a mapping of object to their corresponding `BaseNodes` which will be used to store information
        * a count of `fn`s by type for naming
        * a set of edges, each a pair of objects that are linked by a computation
    """

    def __init__(self):
        self.nodes = OrderedDict()
        self.node_types = dict()
        self.node_set = set()
        self.edges = set()

        self._create_dummy()

    def _create_dummy(self):
        """Construct a dummy node as the context for the recording graph.

        Adds a dummy node (mapped to `None`) to the recording graph as the
        context.  This node has the least depth (-1); is its own parent, and
        will have only one child: the network whose execution is to be
        recorded.

        :returns:   `None`

        """
        self.node_set.add(None)
        self.nodes[None] = BaseNode(fn=None, depth=-1, parent=None, name="ContextDummy")

    def add_node(self, net, depth=0, parent=None, name=None):
        """Construct a node of recording graph.

        Construct a node that will store information related to `net`
        as the neural network is run.

        :net:       Object whose information will be stored (a `torch.Tensor`,
                    `torch.Parameter`, `torch.nn.Module`, a computation op, or `None`).
                    Referred to as `node.fn` object from now on.
        :depth:     The depth at which `net` is found/will be rendered
        :parent:    The object as part of which net will be run
        :name:      a name to recognize the object during rendering, defaults to class name
        :returns:   `None`

        """
        classname = type(net).__name__
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
        """Construct an edge of the recording graph.

        Records an edge between two `node.fn` objects to be used while rendering.
        This will be used along with the `nodes` dictionary to map edges properly.

        :_from: a `node.fn` object
        :_to:   a `node.fn` object
        :returns:

        """
        if _from is None or _to is None:
            raise AssertionError("Cannot draw edge involving" + str((_from, _to)))
        edge = (_from, _to)
        if edge not in self.edges:
            self.edges.add(edge)


def op_acc(gf, rec, node):
    """Operator Accumulator.

    Creates an `OpNode` to record the newly-performed operation `gf`, if not
    already recorded. If `gf` is an initialization op (`AccumulateGradient`),
    then links `gf` to its connected `torch.Tensor` instead of creating an
    `OpNode`. Otherwise recursively checks all operations that are connected to
    `gf` and adds them if necessary.

    :gf:    current operation, a `grad_fn` object obtained from a `torch.Tensor`
    :rec:   a `Recorder` object whose nodes are updated
    :node:  `LayerNode` whose `fn` the current operation is a part of
    :returns:   None

    """
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
    """Tensor Accumulator.

    Creates a `TensorNode` to record the newly-created tensor, if not already
    recorded.  Note that the resulting `TensorNode` has the same parent as
    `node`, because the underlying tensor is the output of/input to `node.fn`.

    :tensor:    a `torch.Tensor`
    :rec:       a `Recorder` object whose nodes are updated
    :node:      a `LayerNode` whose `fn` outputs/inputs `tensor`
    :returns:   None

    """
    if tensor in rec.node_set:
        pass
    else:
        rec.add_node(tensor, depth=node.depth, parent=node.parent)


def param_acc(param, rec, node):
    """Parameter Accumulator.

    Creates a `ParamNode` to record the parameter `param` of `node.fn`, if not
    already recorded.  Note that `node.fn` is the *parent* of the resulting
    `ParamNode`.

    :param:     a `torch.Parameter` object, part of `node.fn`
    :rec:       the `Recorder object whose nodes are updated
    :node:      `LayerNode` whose `fn` contains `param`
    :returns:   None

    """
    if param in rec.node_set:
        pass
    else:
        rec.add_node(param, depth=node.depth + 1, parent=node.fn)


# TODO: This isn't used anymore, remove? <03-02-20, ahgamut> #
def leaf_dummy(tensor, rec):
    """Performs a dummy operation (adding 0) to a leaf tensor.

    This ensures that the operations performed hereafter on `tensor` can be
    correctly mapped to their parent. The dummy tensor (and operation) are not
    recorded separately, they merely point to the original tensor.

    :tensor:    a newly-formed leaf 'torch.Tensor`
    :rec:       the `Recorder` object whose nodes are updated
    :returns:   `tensor` after adding 0

    """
    dummy = tensor + 0
    rec.node_set.add(dummy.grad_fn)
    rec.node_set.add(dummy)
    rec.nodes[dummy] = rec.nodes[tensor]
    rec.nodes[dummy.grad_fn] = rec.nodes[tensor]
    return dummy


def generate_prehook(rec, node):
    """Closure to generate a pre-hook for a `torch.nn.Module`.

    :rec:   a `Recorder` object for global information
    :node:  a `node` which is to be associated with the `module`.
    :returns:   the `prehook` function.

    """

    def prehook(module, inputs):
        """Executed BEFORE the given `torch.nn.Module` is run.

        Records parameters contained in `module`, then checks each tensor in
        `inputs` for any operations that may have run after the end of the
        previous `module`. The `inputs` are then converted to leaf tensors and
        recorded before being passed off to the `module`.

        :module:    a `torch.nn.Module`
        :inputs:    a `torch.Tensor` or a tuple of `torch.Tensor`s
        :returns:   the `leaf`-equivalent of `inputs`.

        """
        for name, param in module.named_parameters(recurse=False):
            param_acc(param, rec, node)
            if name is not None and name != "":
                rec.nodes[param].name = name
        is_singleton = not isinstance(inputs, tuple)
        a = [inputs] if is_singleton else inputs  # same input appearing multiple times?
        new_inputs = []
        for x in a:
            gf = x.grad_fn
            op_acc(gf, rec, rec.nodes[node.parent])
            if not x.is_leaf and x not in rec.node_set:
                x = x.detach()
                x.requires_grad = True
            tensor_acc(x, rec, node)
            if gf is not None:
                rec.add_edge(_from=gf, _to=x)
            new_inputs.append(x)
        return new_inputs[0] if is_singleton else tuple(new_inputs)

    return prehook


def generate_posthook(rec, node):
    """Closure to generate a hook for a `torch.nn.Module`.

    :rec:   a `Recorder` object for global information
    :node:  a `node` which is to be associated with the `module`.
    :returns:   the `prehook` function.

    """

    def posthook(module, inputs, outputs):
        """Executed AFTER the given `torch.nn.Module` has run and returned.

        Records any operations that may have run as part of `module`, then if
        checks each tensor in the `outputs` has already been recorded by a
        sub`module` of the current `module` (the sub`module` would execute
        first!). The `outputs` are then converted to leaf tensors to record
        operations afresh.

        :module:    a `torch.nn.Module`
        :inputs:    a `torch.Tensor` or a tuple of `torch.Tensor`s
        :outputs:   a `torch.Tensor` or a tuple of `torch.Tensor`s
        :returns:   the `leaf`-equivalent of `outputs`.

        """
        is_singleton = not isinstance(outputs, tuple)
        b = [outputs] if is_singleton else outputs
        new_outputs = []
        for x in b:
            gf = x.grad_fn
            if not x.is_leaf and x not in rec.node_set:
                x = x.detach()
                x.requires_grad = True
                tensor_acc(x, rec, node)
            elif x in rec.node_set:
                if node.fn == rec.nodes[x].parent:
                    rec.nodes[x].depth -= 1
                    rec.nodes[x].parent = node.parent
                    node.subnets.remove(rec.nodes[x].fn)
            op_acc(gf, rec, node)
            if gf is not None:
                rec.add_edge(_from=gf, _to=x)
            new_outputs.append(x)
        return new_outputs[0] if is_singleton else tuple(new_outputs)

    return posthook


def generate_backhook(rec, node):
    def backhook(module, grad_inputs, grad_outputs):
        pass

    return backhook
