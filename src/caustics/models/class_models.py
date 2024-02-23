# mypy: disable-error-code="operator"
from typing import Literal, Union, Annotated, Optional, List, Any, Dict
from pydantic import BaseModel, ConfigDict, Field

from ..parametrized import Parametrized
from .registry import get_kind
from .params import (
    Parameters,
    FlatLambdaCDM_Params,
    SIE_Params,
    Sersic_Params,
    Lens_Source_Params,
)


class FileInput(BaseModel):
    path: str = Field(..., description="The path to the file")


class StateDict(BaseModel):
    load: FileInput


class ClassParams(Parameters):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Base(BaseModel):
    name: str = Field(..., description="Name of the object")
    kind: str = Field(..., description="Kind of the object")
    params: Optional[Parameters] = Field(None, description="Parameters of the object")
    class_params: Optional[ClassParams] = Field(
        None, description="Class parameters for object creation"
    )

    # internal
    _cls: Parametrized

    def __init__(self, **data):
        super().__init__(**data)
        self._cls = get_kind(self.kind)

    def get_class_params_dump(self, class_params: ClassParams) -> Dict[str, Any]:
        """
        Get the model dump of the class parameters,
        if the field is a model then get the model object.

        Parameters
        ----------
        class_params : ClassParams
            The class parameters to dump

        Returns
        -------
        dict
            The model dump of the class parameters
        """
        model_dict = {}
        for f in class_params.model_fields_set:
            model = getattr(class_params, f)
            if isinstance(model, Base):
                model_dict[f] = model.model_obj()
            elif isinstance(model, list):
                model_dict[f] = [m.model_obj() for m in model]
            else:
                model_dict[f] = getattr(class_params, f)
        return model_dict

    def model_obj(self) -> Any:
        class_params = (
            self.get_class_params_dump(self.class_params) if self.class_params else {}
        )  # Capture None case
        params = self.params.model_dump() if self.params else {}  # Capture None case
        return self._cls(name=self.name, **class_params, **params)


# Object Models
class FlatLambdaCDM(Base):
    kind: Literal["FlatLambdaCDM"] = "FlatLambdaCDM"
    params: Optional[FlatLambdaCDM_Params] = Field(
        FlatLambdaCDM_Params(), description="Parameters of the object"
    )


cosmology = Annotated[Union[FlatLambdaCDM], Field(discriminator="kind")]


class Lenses_Class_Params(ClassParams):
    cosmology: cosmology


class SIE_Class_Params(Lenses_Class_Params):
    z_l: Optional[float] = Field(None, description="The redshift of the lens")
    s: float = Field(0.0, description="The core radius of the lens")


class SIE(Base):
    kind: Literal["SIE"] = "SIE"
    params: Optional[SIE_Params] = Field(
        SIE_Params(), description="Parameters of the object"
    )
    class_params: Optional[SIE_Class_Params] = Field(
        None, description="Keyword arguments for the object"
    )


single_lenses = Annotated[Union[SIE], Field(discriminator="kind")]


class Multiplane_Class_Params(Lenses_Class_Params):
    lenses: List[single_lenses] = Field([], description="A list of lens objects")


class Multiplane(Base):
    kind: Literal["Multiplane"] = "Multiplane"
    params: None = None
    class_params: Optional[Multiplane_Class_Params] = Field(
        None, description="Keyword arguments for the object"
    )


lenses = Annotated[Union[single_lenses, Multiplane], Field(discriminator="kind")]


class Sersic_Class_Params(ClassParams):
    s: float = Field(0.0, description="A small constant for numerical stability")
    use_lenstronomy_k: bool = Field(
        False,
        description="A flag indicating whether to use lenstronomy to compute the value of k",
    )


class Sersic(Base):
    kind: Literal["Sersic"] = "Sersic"
    params: Optional[Sersic_Params] = Field(
        Sersic_Params(), description="Parameters of the object"
    )
    class_params: Optional[Sersic_Class_Params] = Field(
        None, description="Keyword arguments for the object"
    )


light = Annotated[Union[Sersic], Field(discriminator="kind")]


class Lense_Source_Class_Params(ClassParams):
    lens: lenses
    source: light
    pixelscale: float
    pixels_x: int
    lens_light: Optional[light] = Field(
        None,
        description="caustics light object which defines the lensing object's light",
    )
    psf: Optional[float] = Field(
        None,
        description="An image to convolve with the scene. Note that if ``upsample_factor > 1`` the psf must also be at the higher resolution.",
    )
    pixels_y: Optional[int] = Field(
        None,
        description="number of pixels on the y-axis for the sampling grid. If left as ``None`` then this will simply be equal to ``gridx``",
    )
    upsample_factor: int = 1
    psf_pad: bool = True
    psf_mode: Literal["fft", "conv2d"] = "fft"


class Lens_Source(Base):
    kind: Literal["Lens_Source"] = "Lens_Source"
    params: Optional[Lens_Source_Params] = Field(
        Lens_Source_Params(), description="Parameters of the object"
    )
    class_params: Optional[Lense_Source_Class_Params] = Field(
        None, description="Keyword arguments for the object"
    )
    state: Optional[StateDict] = Field(
        None, description="State safetensor for the simulator"
    )


simulators = Annotated[Union[Lens_Source], Field(discriminator="kind")]

# Config


class Config(BaseModel):
    simulator: simulators
