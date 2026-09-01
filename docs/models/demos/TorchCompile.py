import torch
import torch.nn as nn
import torch.nn.functional as F


def compiled_silu(x):
    return F.silu(x)


class CompiledActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.compile(compiled_silu)

    def forward(self, x):
        return self.activation(x)


class ModelWithCompiledFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = CompiledActivation()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(self.activation(x))


model = ModelWithCompiledFunction()
model.eval()
example_input = torch.randn(2, 16)

code_contents = """\
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvista import trace_model


def compiled_silu(x):
    return F.silu(x)


class CompiledActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.compile(compiled_silu)

    def forward(self, x):
        return self.activation(x)


class ModelWithCompiledFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = CompiledActivation()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(self.activation(x))


model = ModelWithCompiledFunction()
model.eval()
example_input = torch.randn(2, 16)

trace_model(model, example_input)

"""
