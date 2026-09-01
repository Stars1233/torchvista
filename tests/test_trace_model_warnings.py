import warnings
from types import SimpleNamespace

import torch
import torch.nn as nn

import torchvista.tracer as tracer


class ModelWithCompiledCallable(nn.Module):
    def __init__(self):
        super().__init__()

        def original(x):
            return x + 1

        def compiled(x):
            return original(x)

        compiled._torchdynamo_orig_callable = original
        self.activation = compiled

    def forward(self, x):
        return self.activation(x)


def disable_rendering(monkeypatch):
    monkeypatch.setattr(tracer, "process_graph", lambda *args, **kwargs: None)
    monkeypatch.setattr(tracer, "plot_graph", lambda *args, **kwargs: None)


def test_warns_for_compiled_callable_without_force_eager(monkeypatch):
    disable_rendering(monkeypatch)
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        monkeypatch.delattr(compiler, "set_stance", raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracer.trace_model(ModelWithCompiledCallable().eval(), torch.randn(1, 4))

    assert len(caught) == 1
    assert caught[0].category is UserWarning
    assert "torch.compile-wrapped callables" in str(caught[0].message)
    assert "visualization may be incomplete or contain disconnected nodes" in str(
        caught[0].message
    )


def test_does_not_warn_when_force_eager_is_available(monkeypatch):
    disable_rendering(monkeypatch)
    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        compiler = SimpleNamespace()
        monkeypatch.setattr(torch, "compiler", compiler, raising=False)
    monkeypatch.setattr(compiler, "set_stance", lambda stance: None, raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracer.trace_model(ModelWithCompiledCallable().eval(), torch.randn(1, 4))

    assert caught == []


def test_does_not_warn_for_eager_model(monkeypatch):
    disable_rendering(monkeypatch)
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        monkeypatch.delattr(compiler, "set_stance", raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracer.trace_model(nn.Linear(4, 4).eval(), torch.randn(1, 4))

    assert caught == []
