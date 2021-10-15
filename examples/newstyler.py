import sys
import torch
import torchrecord
from torchrecord.renderer import GraphvizStyler


class ConvSample(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=5, kernel_size=5, stride=2, padding=2
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=1
        )
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = x1 + x2
        return self.relu(x3)


class MyStyler(GraphvizStyler):
    def style_node(self, node):
        default = super().style_node(node)
        if isinstance(node.fn, torch.nn.Conv2d):
            params = {}
            params["kernel_size"] = node.fn.kernel_size
            params["padding"] = node.fn.padding
            params["stride"] = node.fn.stride
            default["label"] = (
                node.name
                + "\n("
                + ",\n".join("{}={}".format(k, v) for k, v in params.items())
                + ")"
            )
            default["penwidth"] = "2.4"
        return default

    def style_edge(self, fnode, tnode):
        if isinstance(fnode.fn, torch.nn.Conv2d) and isinstance(tnode.fn, torch.Tensor):
            return {"penwidth": "4.8", "color": "#ee8800"}
        elif isinstance(tnode.fn, torch.nn.ReLU) and isinstance(fnode.fn, torch.Tensor):
            return {"penwidth": "4.8", "color": "#00228f"}
        else:
            return super().style_edge(fnode, tnode)


def main():
    net = ConvSample()
    rec = torchrecord.record(net, name="ConvSample", input_shapes=(1, 1, 10, 10))
    g = torchrecord.make_dot(rec, render_depth=1, styler_cls=MyStyler, fontname="Lato")
    g.format = "svg"
    g.attr(label="Custom Styler Class")
    g.render("{}-{}".format("CustomStyler", 1), directory="./", cleanup=True)


if __name__ == "__main__":
    main()
