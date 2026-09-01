import importlib

import pytest
import torch.nn as nn

import torchvista.module_discovery as module_discovery


OPTIONAL_PACKAGES = {"torchvision", "torchaudio", "torchtext"}


@pytest.mark.parametrize("failing_package", sorted(OPTIONAL_PACKAGES))
def test_broken_optional_package_does_not_abort_discovery(
    failing_package, monkeypatch, capsys
):
    real_import_module = importlib.import_module

    def fake_import_module(package_name):
        if package_name == failing_package:
            raise AttributeError("incompatible package")
        if package_name in OPTIONAL_PACKAGES:
            return None
        return real_import_module(package_name)

    monkeypatch.setattr(module_discovery.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(module_discovery.pkgutil, "iter_modules", lambda *args: [])

    discovered = module_discovery.get_all_nn_modules()

    assert nn.Linear in discovered
    output = capsys.readouterr().out
    assert f"[warning] {failing_package} is available, but its import failed" in output


def test_missing_optional_packages_do_not_emit_failure_warning(monkeypatch, capsys):
    real_import_module = importlib.import_module

    def fake_import_module(package_name):
        if package_name in OPTIONAL_PACKAGES:
            raise ImportError(f"No module named {package_name}")
        return real_import_module(package_name)

    monkeypatch.setattr(module_discovery.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(module_discovery.pkgutil, "iter_modules", lambda *args: [])

    discovered = module_discovery.get_all_nn_modules()

    assert nn.Linear in discovered
    assert capsys.readouterr().out == ""
