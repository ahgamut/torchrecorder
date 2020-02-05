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


def main2():
    i = int(sys.argv[1])
    net = SampleNet()
    # equivalent to calling render_network
    rec = torchrec.record(net, name="Sample Net", input_shapes=(1, 3))
    g = torchrec.make_dot(rec, render_depth=i)
    # g is graphviz.Digraph object
    g.format = "svg"
    g.attr(label="{} at depth={}".format("Sample Net", i))
    g.render("{}-{}".format("Sample Net", i), directory="./", cleanup=True)


if __name__ == "__main__":
    main()
