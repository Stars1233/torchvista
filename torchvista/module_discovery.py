import importlib
import inspect
import pkgutil
import warnings

from .overrides import CONTAINER_MODULES


def _import_optional_package(package_name):
    try:
        return importlib.import_module(package_name)
    except ImportError:
        return None
    except Exception:
        print(
            f"[warning] {package_name} is available, but its import failed, so torchvista "
            f"cannot discover {package_name} modules. If you need {package_name} tracing, "
            f"run `import {package_name}` separately to debug what is wrong."
        )
        return None


def get_all_nn_modules():
    import torch.nn as nn

    torchvision = _import_optional_package("torchvision")
    torchaudio = _import_optional_package("torchaudio")
    torchtext = _import_optional_package("torchtext")

    modules_to_scan = [nn, torchvision, torchaudio, torchtext]

    visited = set()
    module_classes = set()

    def walk_module(mod):
        if mod in visited:
            return
        visited.add(mod)

        try:
            for _, obj in inspect.getmembers(mod):
                if inspect.isclass(obj) and issubclass(obj, nn.Module):
                    module_classes.add(obj)
        except Exception:
            return  # Skip modules that can't be introspected

        # Recursively explore submodules
        if hasattr(mod, '__path__'):
            for _, subname, ispkg in pkgutil.iter_modules(mod.__path__, mod.__name__ + "."):
                try:
                    submod = importlib.import_module(subname)
                    walk_module(submod)
                except Exception:
                    continue  # skip if can't import

    for mod in modules_to_scan:
        if mod is not None:
            walk_module(mod)

    return module_classes


with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    MODULES = get_all_nn_modules() - CONTAINER_MODULES
