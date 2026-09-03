import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv


class DoubleWidth(nn.Module):
    def forward(self, x):
        return torch.cat([x, x], dim=1)


model = DoubleWidth()
model.eval()

shape_env = ShapeEnv()
fake_mode = FakeTensorMode(shape_env=shape_env)
example_input = fake_mode.from_tensor(
    torch.randn(2, 16),
    static_shapes=False,
)

code_contents = """\
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torchvista import trace_model


class DoubleWidth(nn.Module):
    def forward(self, x):
        return torch.cat([x, x], dim=1)


model = DoubleWidth()
model.eval()

shape_env = ShapeEnv()
fake_mode = FakeTensorMode(shape_env=shape_env)
example_input = fake_mode.from_tensor(
    torch.randn(2, 16),
    static_shapes=False,
)

trace_model(model, example_input)

"""
