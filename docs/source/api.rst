API Reference
=============

Convenience Functions
---------------------

.. autofunction:: torchrecord.render_network

.. autofunction:: torchrecord.record

.. autofunction:: torchrecord.make_dot


Custom `graphviz` styling
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchrecord.renderer.GraphvizStyler
    :members:

The `~torchrecord.renderer.GraphvizStyler.style_node` and `~torchrecord.renderer.GraphvizStyler.style_edge` methods read the properties
`~torchrecord.nodes.BaseNode` objects, so any subclass of `~torchrecord.renderer.GraphvizStyler` would need the same.

.. autoclass:: torchrecord.nodes.TensorNode
.. autoclass:: torchrecord.nodes.OpNode
.. autoclass:: torchrecord.nodes.LayerNode
.. autoclass:: torchrecord.nodes.ParamNode
.. autoclass:: torchrecord.nodes.BaseNode

Custom Rendering
----------------

If you are creating a new format to render information from a `~torchrecord.recorder.Recorder`\ , 
you would need to subclass the following methods in `~torchrecord.renderer.base.BaseRenderer`\ , 
as done in `~torchrecord.renderer.GraphvizRenderer`:

* `~torchrecord.renderer.GraphvizRenderer.render_node`
* `~torchrecord.renderer.GraphvizRenderer.render_recursive_node`
* `~torchrecord.renderer.GraphvizRenderer.render_edge`

.. autoclass:: torchrecord.renderer.GraphvizRenderer
    :members:


.. autoclass:: torchrecord.renderer.base.BaseRenderer
    :members:


Custom Recording
----------------

Subclassing `~torchrecord.recorder.Recorder` should be unnecessary in most cases.

.. autoclass:: torchrecord.recorder.Recorder
.. autofunction:: torchrecord.recorder.op_acc
.. autofunction:: torchrecord.recorder.tensor_acc
.. autofunction:: torchrecord.recorder.param_acc
.. autofunction:: torchrecord.recorder.leaf_dummy
.. autofunction:: torchrecord.recorder.prehook
.. autofunction:: torchrecord.recorder.posthook


