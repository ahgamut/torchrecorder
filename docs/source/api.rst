API Reference
=============

Convenience Functions
---------------------

.. autofunction:: torchrecorder.render_network

.. autofunction:: torchrecorder.record

.. autofunction:: torchrecorder.make_dot


Custom `graphviz` styling
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchrecorder.renderer.GraphvizStyler
    :members:

The `~torchrecorder.renderer.GraphvizStyler.style_node` and `~torchrecorder.renderer.GraphvizStyler.style_edge` methods read the properties
`~torchrecorder.nodes.BaseNode` objects, so any subclass of `~torchrecorder.renderer.GraphvizStyler` would need the same.

.. autoclass:: torchrecorder.nodes.TensorNode
.. autoclass:: torchrecorder.nodes.OpNode
.. autoclass:: torchrecorder.nodes.LayerNode
.. autoclass:: torchrecorder.nodes.ParamNode
.. autoclass:: torchrecorder.nodes.BaseNode

Custom Rendering
----------------

If you are creating a new format to render information from a `~torchrecorder.recorder.Recorder`\ , 
you would need to subclass the following methods in `~torchrecorder.renderer.base.BaseRenderer`\ , 
as done in `~torchrecorder.renderer.GraphvizRenderer`:

* `~torchrecorder.renderer.GraphvizRenderer.render_node`
* `~torchrecorder.renderer.GraphvizRenderer.render_recursive_node`
* `~torchrecorder.renderer.GraphvizRenderer.render_edge`

.. autoclass:: torchrecorder.renderer.GraphvizRenderer
    :members:


.. autoclass:: torchrecorder.renderer.base.BaseRenderer
    :members:


Custom Recording
----------------

Subclassing `~torchrecorder.recorder.Recorder` should be unnecessary in most cases.

.. autoclass:: torchrecorder.recorder.Recorder
.. autofunction:: torchrecorder.recorder.op_acc
.. autofunction:: torchrecorder.recorder.tensor_acc
.. autofunction:: torchrecorder.recorder.param_acc
.. autofunction:: torchrecorder.recorder.leaf_dummy
.. autofunction:: torchrecorder.recorder.prehook
.. autofunction:: torchrecorder.recorder.posthook


