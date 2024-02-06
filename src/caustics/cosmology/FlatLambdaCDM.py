# mypy: disable-error-code="operator"
from typing import Optional

import torch
from torch import Tensor
from scipy.special import hyp2f1
from astropy.cosmology import default_cosmology

from ..utils import interp1d
from ..parametrized import unpack
from ..packed import Packed
from ..constants import c_Mpc_s, km_to_Mpc
from .base import Cosmology
from .base import (
    _comoving_distance_z1z2,
    _transverse_comoving_distance_z1z2,
    _angular_diameter_distance,
    _angular_diameter_distance_z1z2,
    _time_delay_distance,
    _critical_surface_density,
)

_h0_default = float(default_cosmology.get().h)
_critical_density_0_default = float(
    default_cosmology.get().critical_density(0).to("solMass/Mpc^3").value
)
_Om0_default = float(default_cosmology.get().Om0)

# Set up interpolator to speed up comoving distance calculations in Lambda-CDM
# cosmologies. Construct with float64 precision.
_comoving_distance_helper_x_grid = 10 ** torch.linspace(-3, 1, 500, dtype=torch.float64)
_comoving_distance_helper_y_grid = torch.as_tensor(
    _comoving_distance_helper_x_grid
    * hyp2f1(1 / 3, 1 / 2, 4 / 3, -(_comoving_distance_helper_x_grid**3)),
    dtype=torch.float64,
)

h0_default = torch.tensor(_h0_default)
critical_density_0_default = torch.tensor(_critical_density_0_default)
Om0_default = torch.tensor(_Om0_default)


def hubble_distance(h0: Tensor):
    """
    Calculate the Hubble distance.

    Parameters
    ----------
    h0: Tensor
        Hubble constant.

    Returns
    -------
    Tensor
        Hubble distance.
    """
    return c_Mpc_s / (100 * km_to_Mpc) / h0


def critical_density(
    z: Tensor,
    Om0: Tensor = Om0_default,
    critical_density_0: Tensor = critical_density_0_default,
) -> Tensor:
    """
    Calculate the critical density at redshift z.

    Parameters
    ----------
    z : Tensor
        Redshift.
    Om0: Tensor, default Om0_default
        Matter density parameter at z=0.
    critical_density_0: Tensor, default critical_density_0_default
        Critical density at z=0.

    Returns
    -------
    Tensor
        _description_
    """

    Ode0 = 1 - Om0
    return critical_density_0 * (Om0 * (1 + z) ** 3 + Ode0)  # fmt: skip


def comoving_distance(
    z: Tensor,
    h0: Tensor = h0_default,
    Om0: Tensor = Om0_default,
) -> Tensor:
    """
    Calculate the comoving distance to redshift z.

    Parameters
    ----------
    z: Tensor
        Redshift
    h0: Tensor, default h0_default
        Hubble constant over 100
    Om0: Tensor, default Om0_default
        Matter density parameter at z=0

    Returns
    -------
    Tensor
        Comoving distance to redshift z.
    """
    # TODO: Figure out how to move these grids
    # to the same device and dtype as the other
    # tensors.
    x_grid = _comoving_distance_helper_x_grid
    y_grid = _comoving_distance_helper_y_grid

    # Helper method for computing comoving distances.
    def _helper(x: Tensor) -> Tensor:
        return interp1d(
            x_grid,
            y_grid,
            torch.atleast_1d(x),
        ).reshape(x.shape)

    Ode0 = 1 - Om0
    ratio = (Om0 / Ode0) ** (1 / 3)
    DH = hubble_distance(h0)
    DC1z = _helper((1 + z) * ratio)
    DC = _helper(ratio)
    return DH * (DC1z - DC) / (Om0 ** (1 / 3) * Ode0 ** (1 / 6))  # fmt: skip


def transverse_comoving_distance(
    z: Tensor,
    h0: Tensor = h0_default,
    Om0: Tensor = Om0_default,
) -> Tensor:
    return comoving_distance(z, h0, Om0)


# Uniform functions in all cosmology classes
def comoving_distance_z1z2(
    z1: Tensor,
    z2: Tensor,
    h0=h0_default,
    Om0=Om0_default,
) -> Tensor:
    return _comoving_distance_z1z2(comoving_distance, z1, z2, h0=h0, Om0=Om0)


def transverse_comoving_distance_z1z2(
    z1: Tensor,
    z2: Tensor,
    h0=h0_default,
    Om0=Om0_default,
):
    return _transverse_comoving_distance_z1z2(
        transverse_comoving_distance, z1, z2, h0=h0, Om0=Om0
    )


def angular_diameter_distance(
    z,
    h0=h0_default,
    Om0=Om0_default,
):
    return _angular_diameter_distance(comoving_distance, z, h0=h0, Om0=Om0)


def angular_diameter_distance_z1z2(
    z1: Tensor, z2: Tensor, h0=h0_default, Om0=Om0_default
):
    return _angular_diameter_distance_z1z2(
        comoving_distance_z1z2, z1, z2, h0=h0, Om0=Om0
    )


def time_delay_distance(
    z_l: Tensor,
    z_s: Tensor,
    h0=h0_default,
    Om0=Om0_default,
):
    return _time_delay_distance(
        angular_diameter_distance,
        angular_diameter_distance_z1z2,
        z_l,
        z_s,
        h0=h0,
        Om0=Om0,
    )


def critical_surface_density(
    z_l: Tensor,
    z_s: Tensor,
    h0=h0_default,
    Om0=Om0_default,
):
    return _critical_surface_density(
        angular_diameter_distance,
        angular_diameter_distance_z1z2,
        z_l,
        z_s,
        h0=h0,
        Om0=Om0,
    )


class FlatLambdaCDM(Cosmology):
    """
    Subclass of Cosmology representing a Flat Lambda Cold Dark Matter (LCDM)
    cosmology with no radiation.
    """

    def __init__(
        self,
        h0: Optional[Tensor] = h0_default,
        critical_density_0: Optional[Tensor] = critical_density_0_default,
        Om0: Optional[Tensor] = Om0_default,
        name: Optional[str] = None,
    ):
        """
        Initialize a new instance of the FlatLambdaCDM class.

        Parameters
        ----------
        name: str
        Name of the cosmology.
        h0: Optional[Tensor]
            Hubble constant over 100. Default is h0_default.
        critical_density_0: (Optional[Tensor])
            Critical density at z=0. Default is critical_density_0_default.
        Om0: Optional[Tensor]
            Matter density parameter at z=0. Default is Om0_default.
        """
        super().__init__(name)

        self.add_param("h0", h0)
        self.add_param("critical_density_0", critical_density_0)
        self.add_param("Om0", Om0)

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        super().to(device, dtype)

    def hubble_distance(self, h0):
        """
        Calculate the Hubble distance.

        Parameters
        ----------
        h0: Tensor
            Hubble constant.

        Returns
        -------
        Tensor
            Hubble distance.
        """
        return hubble_distance(h0)

    @unpack
    def critical_density(
        self,
        z: Tensor,
        *args,
        params: Optional["Packed"] = None,
        h0: Optional[Tensor] = None,
        critical_density_0: Optional[Tensor] = None,
        Om0: Optional[Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Calculate the critical density at redshift z.

        Parameters
        ----------
        z: Tensor
            Redshift.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        torch.Tensor
            Critical density at redshift z.
        """
        return critical_density(z, Om0, critical_density_0)

    @unpack
    def comoving_distance(
        self,
        z: Tensor,
        *args,
        params: Optional["Packed"] = None,
        h0: Optional[Tensor] = None,
        critical_density_0: Optional[Tensor] = None,
        Om0: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Calculate the comoving distance to redshift z.

        Parameters
        ----------
        z: Tensor
            Redshift.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            Comoving distance to redshift z.
        """
        return comoving_distance(z, h0, Om0)

    @unpack
    def transverse_comoving_distance(
        self,
        z: Tensor,
        *args,
        params: Optional["Packed"] = None,
        h0: Optional[Tensor] = None,
        critical_density_0: Optional[Tensor] = None,
        Om0: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        return transverse_comoving_distance(z, h0, Om0)
