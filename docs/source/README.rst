`torchrecord`
==========

|rtd|

`torchrecord` is a Python package that can be used to record the execution graph of a `torch.nn.Module` and use
it to render a visualization of the network structure via `graphviz`. 

Licensed under MIT License.

Installation
------------

Requirements: 

* Python3.6+
* `PyTorch <https://pytorch.org>`_ v1.3 or greater (the ``cpu`` version)
* The `Graphviz <https://graphviz.gitlab.io>`_ library and ``graphviz`` `python package <https://graphviz.readthedocs.io/en/stable/manual.html>`_

Install via ``pip``:

.. code-block:: bash

    $ pip install torchrecord

Simple Example
--------------

.. literalinclude:: ../../examples/sample.py
    :lines: 1-34
    :emphasize-lines: 3, 26-33


+-------------------------+------------------------+
| ``render_depth`` = 1    | ``render_depth`` = 2   |
+=========================+========================+
| |img1|                  | |img2|                 |
+-------------------------+------------------------+

.. |img1| image:: ../../examples/Sample\ Net-1.svg
          :height:  600px
          :width:   300px

.. |img2| image:: ../../examples/Sample\ Net-2.svg
          :height:  600px
          :width:   300px

.. |rtd| image:: https://readthedocs.org/projects/torchrecord/badge/?version=latest&style=flat
