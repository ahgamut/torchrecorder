# -*- coding: utf-8 -*-
"""
    torchrecorder.recorder
    ~~~~~~~~~~~~~~~~~

    Uses hooks to record the traversal of the execution graph

    :copyright: (c) 2019 by Gautham Venkatasubramanian.
    :license: see LICENSE for more details.
"""
from torch.nn import Module
from collections import OrderedDict
from .nodes import BaseNode, TensorNode, ParamNode, OpNode, LayerNode
from functools import partial
import time


class Recorder(object):

    """Record and store execution graph information

    Attributes:
        fn_set (set):         a set of objects ( `~torchrecorder.nodes.BaseNode.fn`\ s) that contain recordable information
        nodes (dict):           a mapping of `~torchrecorder.nodes.BaseNode.fn`\ s
                                to their corresponding `~torchrecorder.nodes.BaseNode`\ s
        fn_types (dict):      a count of `~torchrecorder.nodes.BaseNode.fn`\ s by type for naming
        edges   (set(tuple)):   a set of edges, each a pair of `~torchrecorder.nodes.BaseNode.fn`\ s
    """

    def __init__(self):
        self.nodes = OrderedDict()
        self.fn_types = dict()
        self.fn_set = set()
        self.edges = set()

        self._start_time = None
        self._create_context()

    def add_node(self, net, depth=0, parent=None, name=None):
        """Construct a node of recording graph.

        Construct a `~.nodes.BaseNode` that will store information related to ``net``
        as the neural network is run.

        Args:

            net :       Object whose information will be stored as the ``fn``
                        attribute of a `~.nodes.BaseNode`
            depth :     The scope depth at which ``net`` is found
            parent :    The object as part of which net will be run
            name :      a name to recognize the object during rendering, defaults to class name

        Returns:
            `None`

        """
        classname = type(net).__name__
        if self.fn_types.get(classname):
            self.fn_types[classname] += 1
        else:
            self.fn_types[classname] = 1

        if name is None:
            objname = classname
            if self.fn_types[classname] > 1:
                objname = objname + "-" + str(self.fn_types[classname])
        else:
            objname = name

        if isinstance(net, Module):
            if depth > 0 and name is not None:
                objname = objname + "\n(" + classname + ")"
            x = LayerNode(name=objname, fn=net, parent=parent, depth=depth)

            x.pre = net.register_forward_pre_hook(partial(prehook, rec=self, node=x))
            x.post = net.register_forward_hook(partial(posthook, rec=self, node=x))
            # x.back = net.register_backward_hook(partial(backhook, rec=self, node=x))
        elif "Tensor" in classname:
            x = TensorNode(name=objname, fn=net, parent=parent, depth=depth)
        elif "Parameter" in classname:
            x = ParamNode(name=objname, fn=net, parent=parent, depth=depth)
        elif hasattr(net, "next_functions"):
            x = OpNode(name=objname, fn=net, parent=parent, depth=depth)
        else:
            raise RuntimeError("Cannot create node for " + str(net))

        self.nodes[net] = x
        self.fn_set.add(net)
        if x.parent is not None:
            pnode = self.nodes[x.parent]
            pnode.subnets.add(net)

    def add_dummy(self, dummy, fn):
        """Point to an existing node to assist recording.

        Instead of creating a separate node, the ``dummy`` object is used to point
        to an existing node containing ``fn``. Used for dummy ops and
        ``AccumulateGradient``\ s (see `leaf_dummy` ).

        Args:
            dummy: a dummy `torch.Tensor` or op that should not be recorded
            fn : a recorded object that will be connected to further ops

        """
        self.fn_set.add(dummy)
        self.nodes[dummy] = self.nodes[fn]

    def add_edge(self, _from, _to):
        """Construct an edge of the recording graph.

        Records an edge between two `~torchrecorder.nodes.BaseNode.fn` objects to be used while rendering.
        This will be used along with the ``nodes`` dictionary to map edges properly.

        Args:
            _from (`~torchrecorder.nodes.BaseNode.fn`\ ):
            _to (`~torchrecorder.nodes.BaseNode.fn`\ ):
        """
        if _from is None or _to is None:
            raise AssertionError("Cannot draw edge involving" + str((_from, _to)))
        if self._start_time is not None:
            timestamp = time.time() - self._start_time
        else:
            timestamp = 0
            self._start_time = time.time()
        edge = (_from, _to, round(timestamp, 6))
        self.edges.add(edge)

    def register_hooks(self, net, depth=0, parent=None, name=None):
        """Register the hooks of the `.Recorder` recursively on
        a `torch.nn.Module`\ .

        The hooks registered are `~functools.partial` versions
        of `prehook` and `posthook` corresponding to each node.

        Args:
            net (`~torch.nn.Module`\ ):
            depth (int):
            parent (`torch.nn.Module`\ ): the parent of ``net``
            name (str): name of ``net``

        Returns:
            `None`

        """
        self.add_node(net, depth, parent, name)
        for n, x in net.named_children():
            self.register_hooks(x, depth=depth + 1, parent=net, name=n)

    def remove_hooks(self):
        """Remove hooks from any `~torch.nn.Module`\ s in
        `~torchrecorder.nodes.LayerNode`\ s.

        After the recording is completed, the hooks in
        `~torchrecorder.nodes.LayerNode`\ s are unnecessary.
        They are removed to prevent any possible issues.
        """
        for node in set(self.nodes.values()):
            if isinstance(node, LayerNode):
                node.pre.remove()
                node.post.remove()

    def _create_context(self):
        """Construct a dummy node as the context for the recording graph.

        Adds a dummy node (mapped to `None`) to the recording graph as the
        context.  This node has the least depth (-1); is its own parent, and
        will have only one child: the network whose execution is to be
        recorded.

        Returns:
            `None`
        """
        self.fn_set.add(None)
        self.nodes[None] = BaseNode(fn=None, depth=-1, parent=None, name="ContextDummy")


def op_acc(gf, rec, node):
    """Operator Accumulator.

    Creates an `~.nodes.OpNode` to record the newly-performed operation ``gf``, if not
    already recorded. If ``gf`` is an initialization op (``AccumulateGradient``),
    then points ``gf`` to its connected `torch.Tensor` instead of creating an
    `~.nodes.OpNode`. Otherwise recursively checks all operations that are connected to
    ``gf`` and adds them if necessary.

    Args:
        gf:    current operation, a ``grad_fn`` object obtained from a `torch.Tensor`
        rec:   a `~.Recorder` object whose nodes are updated
        node:  `~.nodes.LayerNode` whose ``fn`` the current operation is a part of

    Returns:
        `None`

    """
    if gf in rec.fn_set:
        pass
    else:
        if hasattr(gf, "variable"):
            rec.add_dummy(dummy=gf, fn=gf.variable)
        elif hasattr(gf, "next_functions"):
            rec.add_node(gf, node.depth + 1, node.fn)
            for x, y in gf.next_functions:
                if x is not None:
                    op_acc(x, rec, node)
                    rec.add_edge(_from=x, _to=gf)


def tensor_acc(tensor, rec, node):
    """Tensor Accumulator.

    Creates a `~.nodes.TensorNode` to record the newly-created tensor, if not already
    recorded.  Note that the resulting `~.nodes.TensorNode` has the same parent as
    ``node``, because the ``tensor`` is the output of/input to ``node.fn``.

    Args:
        tensor:    a `torch.Tensor`
        rec:       a `~.Recorder` object whose nodes are updated
        node:      a `~.nodes.LayerNode` whose ``fn`` outputs/inputs ``tensor``

    Returns:
        `None`

    """
    if tensor in rec.fn_set:
        pass
    else:
        rec.add_node(tensor, depth=node.depth, parent=node.parent)


def param_acc(param, rec, node):
    """Parameter Accumulator.

    Creates a `~.nodes.ParamNode` to record the parameter ``param`` of ``node.fn``, if not
    already recorded.  Note that ``node.fn`` is the *parent* of ``param``\ .

    Args:
        param:     a `~torch.nn.Parameter`
        rec:       the `~.Recorder` object whose nodes are updated
        node:      `~.nodes.LayerNode` whose ``fn`` contains ``param``

    Returns:
	`None`

    """
    if param in rec.fn_set:
        pass
    else:
        rec.add_node(param, depth=node.depth + 1, parent=node.fn)


def leaf_dummy(tensor, rec):
    """Performs a dummy operation (adding 0) to a leaf `~torch.Tensor`.

    This ensures that the (possibly in-place) operations performed on
    ``tensor`` hereafter can be correctly mapped. The dummy tensor (and
    operation) are not recorded separately, they merely point to the original
    tensor.

    Args:
        tensor:    a newly-formed leaf `torch.Tensor`
        rec:       the `~.Recorder` object whose nodes are updated

    Returns:
        ``tensor`` after adding 0

    """
    dummy = tensor + 0
    rec.add_dummy(dummy=dummy, fn=tensor)
    rec.add_dummy(dummy=dummy.grad_fn, fn=tensor)
    return dummy


def prehook(module, inputs, rec, node):
    """hook to record BEFORE the given ``module`` is run.

    Records parameters contained in ``module``, then checks each tensor in
    ``inputs`` for any operations that may have run after the end of the
    previous ``module``. The ``inputs`` are then converted to leaf tensors and
    recorded before being passed off to the ``module``.

    Args:
        module:     a `torch.nn.Module`
        inputs:     a `torch.Tensor` or a `tuple` of `torch.Tensor`\ s
        rec:        a `~.Recorder` object for global information
        node (`~torchrecorder.nodes.LayerNode`\ ):
                    ``node.fn`` is ``module``\ .

    Returns:
        the ``leaf``-equivalent of ``inputs``.

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
        tensor_acc(x, rec, node)
        if gf is not None:
            rec.add_edge(_from=gf, _to=x)
        new_inputs.append(leaf_dummy(x, rec))
    return new_inputs[0] if is_singleton else tuple(new_inputs)


def posthook(module, inputs, outputs, rec, node):
    """hook to record AFTER the given ``module`` has run and returned.

    Records any operations that may have run as part of ``module``\ , then
    checks if each tensor in the ``outputs`` has already been recorded by a
    sub\ ``module`` of the current ``module`` (the sub\ ``module``\ 's
    `posthook` would execute first!). If necessary, the ``outputs`` are
    converted to leaf tensors to record operations afresh.

    Args:
        module:     a `torch.nn.Module`
        inputs:     a `torch.Tensor` or a tuple of `torch.Tensor`\ s
        outputs:    a `torch.Tensor` or a tuple of `torch.Tensor`\ s
        rec:        a `~.Recorder` object for global information
        node (`~torchrecorder.nodes.LayerNode`\ ):
                    ``node.fn`` is ``module``.

    Returns:
        the ``leaf``-equivalent of ``outputs``.

    """
    is_singleton = not isinstance(outputs, tuple)
    b = [outputs] if is_singleton else outputs
    new_outputs = []
    for x in b:
        gf = x.grad_fn
        if gf not in rec.fn_set:
            x = x.detach()
            x.requires_grad = True
            tensor_acc(x, rec, node)
            op_acc(gf, rec, node)
            rec.add_edge(gf, x)
            new_outputs.append(leaf_dummy(x, rec))
        else:
            # if the op has already been recorded
            # it has to be a dummy op
            rec.nodes[gf].parent = node.parent
            rec.nodes[gf].depth -= 1
            node.subnets.remove(rec.nodes[gf].fn)
            new_outputs.append(x)
    return new_outputs[0] if is_singleton else tuple(new_outputs)


def backhook(module, grad_inputs, grad_outputs, rec, node):
    pass
