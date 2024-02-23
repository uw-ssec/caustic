from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from ..cosmology.FlatLambdaCDM import (
    _h0_default,
    _critical_density_0_default,
    _Om0_default,
)


class Parameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FlatLambdaCDM_Params(Parameters):
    h0: float = Field(_h0_default, description="Hubble constant over 100")
    critical_density_0: float = Field(
        _critical_density_0_default, description="Critical density at z=0"
    )
    Om0: float = Field(_Om0_default, description="Matter density parameter at z=0")


class SIE_Params(Parameters):
    x0: Optional[float] = Field(
        None, description="The x-coordinate of the SIE lens's center."
    )
    y0: Optional[float] = Field(
        None, description="The y-coordinate of the SIE lens's center"
    )
    q: Optional[float] = Field(None, description="The axis ratio of the SIE lens")
    phi: Optional[float] = Field(
        None, description="The orientation angle of the SIE lens"
    )
    b: Optional[float] = Field(None, description="The Einstein radius of the SIE lens")


class Sersic_Params(Parameters):
    x0: Optional[float] = Field(
        None, description="The x-coordinate of the Sersic source's center"
    )
    y0: Optional[float] = Field(
        None, description="The y-coordinate of the Sersic source's center"
    )
    q: Optional[float] = Field(None, description="The axis ratio of the Sersic source")
    phi: Optional[float] = Field(
        None, description="The orientation of the Sersic source (position angle)"
    )
    n: Optional[float] = Field(
        None,
        description="The Sersic index, which describes the degree of concentration of the source",
    )
    Re: Optional[float] = Field(
        None, description="The scale length of the Sersic source"
    )
    Ie: Optional[float] = Field(
        None, description="The intensity at the effective radius"
    )


class Lens_Source_Params(Parameters):
    z_s: Optional[float] = Field(None, description="The redshift of the source")
