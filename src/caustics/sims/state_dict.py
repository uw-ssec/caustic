from datetime import datetime as dt
from collections import OrderedDict
from typing import Any, Dict, Optional
from pathlib import Path

from torch import Tensor
from .._version import __version__
from ..namespace_dict import NamespaceDict, NestedNamespaceDict
from .. import io

from safetensors.torch import save

IMMUTABLE_ERR = TypeError("'StateDict' cannot be modified after creation.")
STATIC_PARAMS = "static"


class ImmutableODict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._created = True

    def __delitem__(self, _) -> None:
        raise IMMUTABLE_ERR

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, "_created"):
            raise IMMUTABLE_ERR
        super().__setitem__(key, value)

    def __setattr__(self, name, value) -> None:
        if hasattr(self, "_created"):
            raise IMMUTABLE_ERR
        return super().__setattr__(name, value)


class StateDict(ImmutableODict):
    """A dictionary object that is immutable after creation.
    This is used to store the parameters of a simulator at a given
    point in time.

    Methods
    -------
    to_params()
        Convert the state dict to a dictionary of parameters.
    """

    __slots__ = ("_metadata", "_created", "_created_time")

    def __init__(self, *args, **kwargs):
        # Get created time
        self._created_time = dt.now()
        # Create metadata
        metadata = {
            "software_version": __version__,
            "created_time": self._created_time.isoformat(),
        }
        # Set metadata
        self._metadata = ImmutableODict(metadata)

        # Now create the object, this will set _created
        # to True, and prevent any further modification
        super().__init__(*args, **kwargs)

    @classmethod
    def from_params(cls, params: "NestedNamespaceDict | NamespaceDict"):
        """Class method to create a StateDict
        from a dictionary of parameters

        Parameters
        ----------
        params : NamespaceDict
            A dictionary of parameters,
            can either be the full parameters
            that are "static" and "dynamic",
            or "static" only.

        Returns
        -------
        StateDict
            A state dictionary object
        """
        if isinstance(params, NestedNamespaceDict) and STATIC_PARAMS in params:
            params: NamespaceDict = params[STATIC_PARAMS].flatten()
        tensors_dict: Dict[str, Tensor] = {k: v.value for k, v in params.items()}
        return cls(tensors_dict)

    def to_params(self) -> NamespaceDict:
        """
        Convert the state dict to a dictionary of parameters.

        Returns
        -------
        NamespaceDict
            A dictionary of 'static' parameters.
        """
        from ..parameter import Parameter

        params = NamespaceDict()
        for k, v in self.items():
            params[k] = Parameter(v)
        return params

    def save(self, file_path: Optional[str] = None) -> str:
        """
        Saves the state dictionary to an optional
        ``file_path`` as safetensors format.
        If ``file_path`` is not given,
        this will default to a file in
        the current working directory.

        *Note: The path specified must
        have a '.st' extension.*

        Parameters
        ----------
        file_path : str, optional
            The file path to save the
            state dictionary to, by default None

        Returns
        -------
        str
            The final path of the saved file
        """
        if not file_path:
            file_path = Path(".") / self.__st_file
        elif isinstance(file_path, str):
            file_path = Path(file_path)

        ext = ".st"
        if file_path.suffix != ext:
            raise ValueError(f"File must have '{ext}' extension")

        return io.to_file(file_path, self._to_safetensors())

    @property
    def __st_file(self) -> str:
        file_format = "%Y%m%dT%H%M%S_caustics.st"
        return self._created_time.strftime(file_format)

    def _to_safetensors(self) -> bytes:
        return save(self, metadata=self._metadata)
