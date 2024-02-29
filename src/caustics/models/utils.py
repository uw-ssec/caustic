# mypy: disable-error-code="union-attr, valid-type, has-type, assignment, arg-type, dict-item, return-value, misc"
from typing import List, Literal, Dict, Annotated, Union, Optional
import inspect
from pydantic import Field, create_model

from ..parametrized import Parametrized
from .base_models import Base, Parameters, InitKwargs
from .registry import get_kind, _registry


def create_pydantic_model(
    cls: "Parametrized | str", dependant_models: Dict[str, type] = {}
) -> Base:
    """
    Create a pydantic model from a Parametrized class.

    Parameters
    ----------
    cls : Parametrized | str
        The Parametrized class to create the model from.
    dependant_models : Dict[str, type], optional
        The dependent models to use, by default {}
        See: https://docs.pydantic.dev/latest/concepts/unions/#nested-discriminated-unions

    Returns
    -------
    Base
        The pydantic model of the Parametrized class.
    """
    if isinstance(cls, str):
        parametrized_class = get_kind(cls)  # type: ignore

    cls_signature = inspect.signature(parametrized_class)  # type: ignore

    field_definitions = {
        "kind": (Literal[parametrized_class.__name__], Field(parametrized_class.__name__))  # type: ignore
    }

    # Setup params model
    # Default the values to None so that it can be inputted later
    meta_params = {}
    for k, v in parametrized_class._meta_params.items():
        default = None
        cls_param = cls_signature.parameters.get(k, None)
        if cls_param is not None:
            param_default = cls_param.default
            # Cast to float... even for Tensor since this is
            # on the pydantic side
            default = param_default if param_default is None else float(param_default)

        meta_params[k] = (
            Optional[float],
            Field(default=default, description=v.get("description")),
        )
    if meta_params:
        params_model = create_model(
            f"{parametrized_class.__name__}_Params", __base__=Parameters, **meta_params
        )
        field_definitions["params"] = (
            params_model,
            Field(params_model(), description="Parameters of the object"),
        )

    # Setup init_kwargs model
    init_kwargs = {}
    for k, v in cls_signature.parameters.items():
        if (k not in parametrized_class._meta_params) and (k != "name"):
            if k in dependant_models:
                dependant_model = dependant_models[k]
                if isinstance(dependant_model, list):
                    # For the multi lens case
                    # dependent model is wrapped in a list
                    dependant_model = dependant_model[0]
                    init_kwargs[k] = (List[dependant_model], Field([]))
                else:
                    init_kwargs[k] = (dependant_model, Field(...))
            elif v.default == inspect._empty:
                init_kwargs[k] = (v.annotation, Field(...))
            else:
                init_kwargs[k] = (v.annotation, Field(v.default))

    if init_kwargs:
        init_model = create_model(
            f"{parametrized_class.__name__}_Init_Kwargs",
            __base__=InitKwargs,
            **init_kwargs,
        )
        field_definitions["init_kwargs"] = (
            init_model,
            Field(None, description="Initiation keyword arguments of the object"),
        )

    model = create_model(
        parametrized_class.__name__, __base__=Base, **field_definitions
    )
    model = model._set_class(parametrized_class)
    return model


def setup_pydantic_models() -> type:
    # Cosmology
    cosmology_models = [create_pydantic_model(cosmo) for cosmo in _registry.cosmology]
    cosmology = Annotated[Union[tuple(cosmology_models)], Field(discriminator="kind")]
    # Light
    light_models = [create_pydantic_model(light) for light in _registry.light]
    light_sources = Annotated[Union[tuple(light_models)], Field(discriminator="kind")]
    # Single Lens
    lens_dependant_models = {"cosmology": cosmology}
    single_lens_models = [
        create_pydantic_model(lens, dependant_models=lens_dependant_models)
        for lens in _registry.single_lenses
    ]
    single_lenses = Annotated[
        Union[tuple(single_lens_models)], Field(discriminator="kind")
    ]
    # Multi Lens
    multi_lens_models = [
        create_pydantic_model(
            lens, dependant_models={"lenses": [single_lenses], **lens_dependant_models}
        )
        for lens in _registry.multi_lenses
    ]
    lenses = Annotated[
        Union[tuple([*single_lens_models, *multi_lens_models])],
        Field(discriminator="kind"),
    ]
    return light_sources, lenses


def setup_simulator_models() -> type:
    light_sources, lenses = setup_pydantic_models()
    # Hard code the dependants for now
    # there's currently only one simulator
    # in the system.
    dependents = {
        "Lens_Source": {
            "source": light_sources,
            "lens_light": light_sources,
            "lens": lenses,
        }
    }
    simulators_models = [
        create_pydantic_model(sim, dependant_models=dependents.get(sim))
        for sim in _registry.simulators
    ]
    return Annotated[Union[tuple(simulators_models)], Field(discriminator="kind")]
