API Reference
=============

Convenience Functions
---------------------

.. autofunction:: torchrec.render_network

.. autofunction:: torchrec.record

.. autofunction:: torchrec.make_dot


Custom `graphviz` styling
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: torchrec.renderer.GraphvizStyler
    :members:

The `~torchrec.renderer.GraphvizStyler.style_node` and `~torchrec.renderer.GraphvizStyler.style_edge` methods read the properties
`~torchrec.nodes.BaseNode` objects, so any subclass of `~torchrec.renderer.GraphvizStyler` would need the same.

.. autoclass:: torchrec.nodes.TensorNode
.. autoclass:: torchrec.nodes.OpNode
.. autoclass:: torchrec.nodes.LayerNode
.. autoclass:: torchrec.nodes.ParamNode
.. autoclass:: torchrec.nodes.BaseNode

Custom Rendering
----------------

If you are creating a new format to render information from a `~torchrec.recorder.Recorder`\ , 
you would need to subclass the following methods in `~torchrec.renderer.base.BaseRenderer`\ , 
as done in `~torchrec.renderer.GraphvizRenderer`:

* `~torchrec.renderer.GraphvizRenderer.render_node`
* `~torchrec.renderer.GraphvizRenderer.render_recursive_node`
* `~torchrec.renderer.GraphvizRenderer.render_edge`

.. autoclass:: torchrec.renderer.GraphvizRenderer
    :members:


.. autoclass:: torchrec.renderer.base.BaseRenderer
    :members:


Custom Recording
----------------

Subclassing `~torchrec.recorder.Recorder` should be unnecessary in most cases.

.. autoclass:: torchrec.recorder.Recorder
.. autofunction:: torchrec.recorder.op_acc
.. autofunction:: torchrec.recorder.tensor_acc
.. autofunction:: torchrec.recorder.param_acc
.. autofunction:: torchrec.recorder.leaf_dummy
.. autofunction:: torchrec.recorder.prehook
.. autofunction:: torchrec.recorder.posthook


