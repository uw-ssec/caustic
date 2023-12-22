from typing import List
from collections import deque
from types import MappingProxyType
import json

from torch import Tensor

from .._version import __version__

from ..parametrized import Parametrized
from ..namespace_dict import NamespaceDict

__all__ = ("Simulator",)


class Simulator(Parametrized):
    """A caustics simulator using Parametrized framework.

    Defines a simulator class which is a callable function that
    operates on the Parametrized framework. Users define the `forward`
    method which takes as its first argument an object which can be
    packed, all other args and kwargs are simply passed to the forward
    method.

    See `Parametrized` for details on how to add/access parameters.

    """

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            packed_args = self.pack(args[0])
            rest_args = args[1:]
        else:
            packed_args = self.pack()
            rest_args = tuple()

        return self.forward(packed_args, *rest_args, **kwargs)

    def state_dict(self) -> MappingProxyType[str, "Tensor | str"]:
        # Get static values only
        static_params = self.params["static"].flatten()

        # Extract the tensors only to dictionary
        tensors = {k: v.value for k, v in static_params.items()}

        metadata = {
            "key_maps": self._key_maps,
            "class_maps": self._module_classes,
            "module_order": _get_traversal_order(self),
        }

        # Return a read-only dictionary
        # the formatting is based on safetensors by huggingface
        # See: https://github.com/huggingface/safetensors#format
        state_dict = MappingProxyType(
            {
                "params": tensors,
                "_metadata": {
                    "state_metadata": json.dumps(metadata),
                    "software_version": __version__,
                },
            }
        )
        return state_dict

    @property
    def _module_classes(self) -> NamespaceDict[str, "Parametrized"]:
        # Only catch modules with static parameters
        modules = (
            NamespaceDict()
        )  # todo make this an ordinary dict and reorder at the end.

        def _get_childs(module):
            # Start from root, and move down the DAG
            modules[module.name] = {
                "module": module.__module__,
                "class": module.__class__.__name__,
            }
            if module._childs != {}:
                for child in module._childs.values():
                    _get_childs(child)

        _get_childs(self)
        # TODO reorder
        return modules


def _get_traversal_order(module) -> List[str]:
    """Traverse the module DAG in reverse topological order.

    Parameters
    ----------
    module : Parametrized
        The module to traverse

    Returns
    -------
    List[str]
        The list of module names
    """
    q = deque()
    q.append(module)
    tq = deque()
    while q:
        node = q.popleft()
        if node is None:
            continue

        tq.appendleft(node.name)

        if node.children:
            for child in node.children:
                q.append(child)

    return list(tq)
