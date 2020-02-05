`torchrec`
==========

![](https://readthedocs.org/projects/pytorchrec/badge/?version=latest&style=flat)

A small package to record execution graphs of neural networks in PyTorch.
The package uses hooks and the `grad_fn` attribute to record information.  
This can be used to generate visualizations at different scope depths. 

Licensed under MIT License.
View documentation at https://pytorchrec.readthedocs.io/

## Installation

Requirements:

* Python3.6+
* [PyTorch](https://pytorch.org) v1.3 or greater (the `cpu` version)
* The [Graphviz](https://graphviz.gitlab.io) library and `graphviz` [python package](https://graphviz.readthedocs.io/en/stable/manual.html).


Install this package:

```
$ pip install torchrec
```

## Usage

Consider the below example network:

```python
import sys
import torch
import torchrec


class SampleNet(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.linear_1 = torch.nn.Linear(in_features=3, out_features=3, bias=True)
        self.linear_2 = torch.nn.Linear(in_features=3, out_features=3, bias=True)
        self.linear_3 = torch.nn.Linear(in_features=6, out_features=1, bias=True)
        self.my_special_relu = torch.nn.ReLU()

    def forward(self, inputs):
        x = self.linear_1(inputs)
        y = self.linear_2(inputs)
        z = torch.cat([x, y], dim=1)
        z = self.my_special_relu(self.linear_3(z))
        return z


def main():
    i = int(sys.argv[1])
    net = SampleNet()
    torchrec.render_network(
        net,
        name="Sample Net",
        input_shapes=(1, 3),
        directory="./",
        fmt="svg",
        render_depth=i,
    )


if __name__ == "__main__":
    main()

```

And visualizations like these can be produced:

<img src="./examples/Sample Net-1.svg" width=200 height=650>


<img src="./examples/Sample Net-2.svg" width=350 height=800>


## Acknowledgements

This is inspired from [`szagoruyko/pytorchviz`](https://github.com/szagoruyko/pytorchviz).  This package
differs from `pytorchviz` as it provides rendering at multiple depths.

Note that for rendering a network via TensorBoard during training, you can use
[`torch.utils.tensorboard.SummaryWriter.add_graph`](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_graph),
which records and renders to a `protobuf` in a single step.  The intended usage of `pytorchrec` is for
presentation purposes.
