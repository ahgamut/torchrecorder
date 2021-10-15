import sys
import torch
import torchrecord


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
    net = SampleNet()
    rec = torchrecord.record(net, name="Sample Net", input_shapes=(1, 3))
    g = torchrecord.make_dot(rec, render_depth=1, fontname="Lato")
    g.format = "svg"
    g.attr(label="Font Change via styler_args")
    g.render("{}-{}".format("StyleArgs", 1), directory="./", cleanup=True)


if __name__ == "__main__":
    main()
