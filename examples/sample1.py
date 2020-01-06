import sys
import torch
from torch import nn
import torchrec


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
    i = int(sys.argv[1])
    net = SampleNet().cpu()
    rec = torchrec.record(net, input_shapes=(1, 3), name="Sample Net")
    g = torchrec.make_dot(rec, render_depth=i)
    # g is a graphviz.Digraph() object
    g.format = "svg"
    g.attr(label="Sample Net at depth={i}".format(i=i))
    g.render("sample1-{i}".format(i=i), directory=".", cleanup=True)


if __name__ == "__main__":
    main()
