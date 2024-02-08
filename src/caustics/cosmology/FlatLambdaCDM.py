# mypy: disable-error-code="operator"
from typing import Optional

import torch
from torch import Tensor
from astropy.cosmology import default_cosmology
from scipy.special import hyp2f1

from ..utils import interp1d
from ..parametrized import unpack
from ..packed import Packed
from ..constants import c_Mpc_s, km_to_Mpc
from .base import (
    Cosmology,
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


def hubble_distance(h0):
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
        h0: Tensor, optional
            Hubble constant over 100. Default is h0_default.
        critical_density_0: Tensor, optional
            Critical density at z=0. Default is critical_density_0_default.
        Om0: Tensor, optional
            Matter density parameter at z=0. Default is Om0_default.
        """
        super().__init__(name)

        self.add_param("h0", h0)
        self.add_param("critical_density_0", critical_density_0)
        self.add_param("Om0", Om0)

        self._comoving_distance_helper_x_grid = _comoving_distance_helper_x_grid.to(
            dtype=torch.float32
        )
        self._comoving_distance_helper_y_grid = _comoving_distance_helper_y_grid.to(
            dtype=torch.float32
        )

    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        super().to(device, dtype)
        self._comoving_distance_helper_y_grid = (
            self._comoving_distance_helper_y_grid.to(device, dtype)
        )
        self._comoving_distance_helper_x_grid = (
            self._comoving_distance_helper_x_grid.to(device, dtype)
        )

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
        Ode0 = 1 - Om0
        return critical_density_0 * (Om0 * (1 + z) ** 3 + Ode0)  # fmt: skip

    @unpack
    def _comoving_distance_helper(
        self, x: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Helper method for computing comoving distances.

        Parameters
        ----------
        x: Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Computed comoving distances.
        """
        return interp1d(
            self._comoving_distance_helper_x_grid,
            self._comoving_distance_helper_y_grid,
            torch.atleast_1d(x),
        ).reshape(x.shape)

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
        Ode0 = 1 - Om0
        ratio = (Om0 / Ode0) ** (1 / 3)
        DH = hubble_distance(h0)
        DC1z = self._comoving_distance_helper((1 + z) * ratio, params)
        DC = self._comoving_distance_helper(ratio, params)
        return DH * (DC1z - DC) / (Om0 ** (1 / 3) * Ode0 ** (1 / 6))  # fmt: skip

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
        return self.comoving_distance(z, params, **kwargs)


_func = FlatLambdaCDM(h0=None, critical_density_0=None, Om0=None)


def comoving_distance(
    z: Tensor,
    h0: Tensor = h0_default,
    critical_density_0: Tensor = critical_density_0_default,
    Om0: Tensor = Om0_default,
) -> Tensor:
    """
    Calculate the comoving distance to redshift z.

    Parameters
    ----------
    z : Tensor
        Redshift.
    h0 : Tensor, optional
        Hubble constant, by default h0_default
    critical_density_0 : Tensor, optional
        Critical density at z=0, by default critical_density_0_default
    Om0 : Tensor, optional
        Matter density parameter at z=0, by default Om0_default

    Returns
    -------
    Tensor
        Comoving distance to redshift z.
    """
    return _func.comoving_distance(
        z, h0=h0, critical_density_0=critical_density_0, Om0=Om0
    )


def critical_density(
    z: Tensor,
    critical_density_0: Tensor = critical_density_0_default,
    Om0: Tensor = Om0_default,
) -> Tensor:
    """
    Calculate the critical density at redshift z.

    Parameters
    ----------
    z : Tensor
        Redshift.
    critical_density_0 : Tensor, optional
        Critical density at z=0, by default critical_density_0_default
    Om0 : Tensor, optional
        Matter density parameter at z=0, by default Om0_default

    Returns
    -------
    Tensor
        Critical density at redshift z.
    """
    return _func.critical_density(
        z, h0=h0_default, critical_density_0=critical_density_0, Om0=Om0
    )


def transverse_comoving_distance(
    z: Tensor,
    h0: Tensor = h0_default,
    critical_density_0: Tensor = critical_density_0_default,
    Om0: Tensor = Om0_default,
) -> Tensor:
    """
    Calculate the transverse comoving distance to redshift z.

    Parameters
    ----------
    z : Tensor
        Redshift.
    h0 : Tensor, optional
        Hubble constant, by default h0_default
    critical_density_0 : Tensor, optional
        Critical density at z=0, by default critical_density_0_default
    Om0 : Tensor, optional
        Matter density parameter at z=0, by default Om0_default

    Returns
    -------
    Tensor
        Transverse comoving distance to redshift z.
    """
    return _func.transverse_comoving_distance(
        z, h0=h0, critical_density_0=critical_density_0, Om0=Om0
    )


def comoving_distance_z1z2(
    z1: Tensor,
    z2: Tensor,
    h0: Tensor = h0_default,
    critical_density_0: Tensor = critical_density_0_default,
    Om0: Tensor = Om0_default,
) -> Tensor:
    """
    Calculate the comoving distance between redshift z1 and z2.

    Parameters
    ----------
    z1 : Tensor
        Lower redshift.
    z2 : Tensor
        Upper redshift.
    h0 : Tensor, optional
        Hubble constant, by default h0_default
    critical_density_0 : Tensor, optional
        Critical density at z=0, by default critical_density_0_default
    Om0 : Tensor, optional
        Matter density parameter at z=0, by default Om0_default

    Returns
    -------
    Tensor
        Comoving distance between redshift z1 and z2.
    """
    return _func.comoving_distance_z1z2(
        z1=z1, z2=z2, h0=h0, critical_density_0=critical_density_0, Om0=Om0
    )


def transverse_comoving_distance_z1z2(
    z1: Tensor,
    z2: Tensor,
    h0: Tensor = h0_default,
    critical_density_0: Tensor = critical_density_0_default,
    Om0: Tensor = Om0_default,
) -> Tensor:
    """
    Calculate the transverse comoving distance between redshift z1 and z2.

    Parameters
    ----------
    z1 : Tensor
        The starting redshifts.
    z2 : Tensor
        The ending redshifts.
    h0 : Tensor, optional
        Hubble constant, by default h0_default
    critical_density_0 : Tensor, optional
        Critical density at z=0, by default critical_density_0_default
    Om0 : Tensor, optional
        Matter density parameter at z=0, by default Om0_default

    Returns
    -------
    Tensor
        The transverse comoving distance between redshift z1 and z2.
    """
    return _func.transverse_comoving_distance_z1z2(
        z1=z1, z2=z2, h0=h0, critical_density_0=critical_density_0, Om0=Om0
    )


def angular_diameter_distance(
    z: Tensor,
    h0: Tensor = h0_default,
    critical_density_0: Tensor = critical_density_0_default,
    Om0: Tensor = Om0_default,
) -> Tensor:
    """
    Compute the angular diameter distance to redshift z.

    Parameters
    ----------
    z : Tensor
        Redshift.
    h0 : Tensor, optional
        Hubble constant, by default h0_default
    critical_density_0 : Tensor, optional
        Critical density at z=0, by default critical_density_0_default
    Om0 : Tensor, optional
        Matter density parameter at z=0, by default Om0_default

    Returns
    -------
    Tensor
        Angular diameter distance to redshift z.
    """
    return _func.angular_diameter_distance(
        z=z, h0=h0, critical_density_0=critical_density_0, Om0=Om0
    )


def angular_diameter_distance_z1z2(
    z1: Tensor,
    z2: Tensor,
    h0: Tensor = h0_default,
    critical_density_0: Tensor = critical_density_0_default,
    Om0: Tensor = Om0_default,
) -> Tensor:
    """
    Compute the angular diameter distance between two redshifts.

    Parameters
    ----------
    z1 : Tensor
        The starting redshifts.
    z2 : Tensor
        The ending redshifts.
    h0 : Tensor, optional
        Huble constant, by default h0_default
    critical_density_0 : Tensor, optional
        Critical density at z=0, by default critical_density_0_default
    Om0 : Tensor, optional
        Matter density parameter at z=0, by default Om0_default
    """
    return _func.angular_diameter_distance_z1z2(
        z1=z1, z2=z2, h0=h0, critical_density_0=critical_density_0, Om0=Om0
    )


def time_delay_distance(
    z_l: Tensor,
    z_s: Tensor,
    h0: Tensor = h0_default,
    critical_density_0: Tensor = critical_density_0_default,
    Om0: Tensor = Om0_default,
) -> Tensor:
    """
    Compute the time delay distance between lens and source planes.

    Parameters
    ----------
    z_l : Tensor
        The lens redshifts.
    z_s : Tensor
        The source redshifts.
    h0 : Tensor, optional
        Hubble constant, by default h0_default
    critical_density_0 : Tensor, optional
        Critical density at z=0, by default critical_density_0_default
    Om0 : Tensor, optional
        Matter density parameter at z=0, by default Om0_default

    Returns
    -------
    Tensor
        The time delay distance for each pair of lens and source redshifts.
    """
    return _func.time_delay_distance(
        z_l=z_l, z_s=z_s, h0=h0, critical_density_0=critical_density_0, Om0=Om0
    )


def critical_surface_density(
    z_l: Tensor,
    z_s: Tensor,
    h0: Tensor = h0_default,
    critical_density_0: Tensor = critical_density_0_default,
    Om0: Tensor = Om0_default,
) -> Tensor:
    """
    Compute the critical surface density for lens and source planes.

    Parameters
    ----------
    z_l : Tensor
        The lens redshifts.
    z_s : Tensor
        The source redshifts.
    h0 : Tensor, optional
        Hubble constant, by default h0_default
    critical_density_0 : Tensor, optional
        Critical density at z=0, by default critical_density_0_default
    Om0 : Tensor, optional
        Matter density parameter at z=0, by default Om0_default

    Returns
    -------
    Tensor
        The critical surface density for each pair of lens and source redshifts.
    """
    return _func.critical_surface_density(
        z_l=z_l, z_s=z_s, h0=h0, critical_density_0=critical_density_0, Om0=Om0
    )
