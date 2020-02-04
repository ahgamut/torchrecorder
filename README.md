PyTorchRec
==========

A small package to record execution graphs of neural networks in PyTorch.
The package uses hooks and the `grad_fn` attribute to record information.  
This can be used to generate visualizations at different resolution depths. 

Licensed under MIT License.

## Installation

(Tested on `torch 1.3.0+cpu`).  
Install [graphviz](https://graphviz.gitlab.io/) for your system.

Install this package:

```
pip install git+https://github.com/ahgamut/pytorchrec
```

## Usage

Consider the below example network:

```python
import sys
from torchrec import record, make_dot
from torch import nn

class SampleNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.linear_1 = nn.Linear(in_features=3, out_features=3, bias=True)
        self.linear_2 = nn.Linear(in_features=3, out_features=3, bias=True)
        self.linear_3 = nn.Linear(in_features=6, out_features=1, bias=True)
        self.my_special_relu = nn.ReLU()

    def forward(self, inputs):
        x = self.linear_1(inputs)
        y = self.linear_2(inputs)
        z = torch.cat([x, y], dim=1)
        z = self.my_special_relu(self.linear_3(z))
        return z

def main():
    net = SampleNet().cpu()
    i = int(sys.argv[1])
    rec = torchrec.record(net, input_shapes=(1, 3), name="Sample Net")
    g = torchrec.make_dot(rec, render_depth=i)
    # g is a graphviz.Digraph() object
    g.format = "svg"
    g.attr(label="Sample Net at depth={i}".format(i=i))
    g.render("sample-{i}".format(i=i), directory=".", cleanup=True)


if __name__ == "__main__":
    main()

```

And visualizations like these can be produced:

<img src="./examples/sample1-1.svg" width=200 height=650>


<img src="./examples/sample1-2.svg" width=350 height=800>


## Acknowledgements

This is inspired from [`szagoruyko/pytorchviz`](https://github.com/szagoruyko/pytorchviz).  This package
differs from `pytorchviz` as it provides rendering at multiple depths.

Note that for rendering a network via TensorBoard during training, you can use
[`torch.utils.tensorboard.SummaryWriter.add_graph`](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_graph),
which records and renders to a `protobuf` in a single step.  The intended usage of `pytorchrec` is for
presentation purposes.
