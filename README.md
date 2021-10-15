`torchrecord`
=============

![](https://readthedocs.org/projects/torchrecord/badge/?version=latest&style=flat)

A small package to record execution graphs of neural networks in PyTorch.
The package uses hooks and the `grad_fn` attribute to record information.  
This can be used to generate visualizations at different scope depths. 

Licensed under MIT License.
View documentation at https://torchrecord.readthedocs.io/

## Installation

Requirements:

* Python3.6+
* [PyTorch](https://pytorch.org) v1.3 or greater (the `cpu` version)
* The [Graphviz](https://graphviz.gitlab.io) library and `graphviz` [python package](https://graphviz.readthedocs.io/en/stable/manual.html).


Install this package:

```
$ pip install torchrecord
```

## Acknowledgements

This is inspired from [`szagoruyko/pytorchviz`](https://github.com/szagoruyko/pytorchviz).  This package
differs from `pytorchviz` as it provides rendering at multiple depths.

Note that for rendering a network during training, you can use TensorBoard and
[`torch.utils.tensorboard.SummaryWriter.add_graph`](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_graph),
which records and renders to a `protobuf` in a single step.  The intended usage of `torchrecord` is for
presentation purposes.
