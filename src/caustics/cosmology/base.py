# mypy: disable-error-code="operator"
from abc import abstractmethod
from math import pi
from typing import Callable, Optional

from torch import Tensor

from ..constants import G_over_c2
from ..parametrized import Parametrized, unpack
from ..packed import Packed


def _comoving_distance_z1z2(
    comoving_distance: Callable, z1: Tensor, z2: Tensor, **kwargs
) -> Tensor:
    return comoving_distance(z2, **kwargs) - comoving_distance(z1, **kwargs)


def _transverse_comoving_distance_z1z2(
    transverse_comoving_distance: Callable, z1: Tensor, z2: Tensor, **kwargs
):
    return transverse_comoving_distance(z2, **kwargs) - transverse_comoving_distance(
        z1, **kwargs
    )


def _angular_diameter_distance(
    comoving_distance: Callable, z: Tensor, **kwargs
) -> Tensor:
    return comoving_distance(z, **kwargs) / (1 + z)


def _angular_diameter_distance_z1z2(
    comoving_distance_z1z2: Callable, z1: Tensor, z2: Tensor, **kwargs
) -> Tensor:
    return comoving_distance_z1z2(z1, z2, **kwargs) / (1 + z2)


def _calculate_angular_diameter_distances(
    angular_diameter_distance: Callable,
    angular_diameter_distance_z1z2: Callable,
    z_l: Tensor,
    z_s: Tensor,
    **kwargs,
):
    d_l = angular_diameter_distance(z_l, **kwargs)
    d_s = angular_diameter_distance(z_s, **kwargs)
    d_ls = angular_diameter_distance_z1z2(z_l, z_s, **kwargs)
    return d_l, d_s, d_ls


def _time_delay_distance(
    angular_diameter_distance: Callable,
    angular_diameter_distance_z1z2: Callable,
    z_l: Tensor,
    z_s: Tensor,
    **kwargs,
):
    d_l, d_s, d_ls = _calculate_angular_diameter_distances(
        angular_diameter_distance, angular_diameter_distance_z1z2, z_l, z_s, **kwargs
    )
    return (1 + z_l) * d_l * d_s / d_ls


def _critical_surface_density(
    angular_diameter_distance: Callable,
    angular_diameter_distance_z1z2: Callable,
    z_l: Tensor,
    z_s: Tensor,
    **kwargs,
):
    d_l, d_s, d_ls = _calculate_angular_diameter_distances(
        angular_diameter_distance, angular_diameter_distance_z1z2, z_l, z_s, **kwargs
    )
    return d_s / (4 * pi * G_over_c2 * d_l * d_ls)  # fmt: skip


class Cosmology(Parametrized):
    """
    Abstract base class for cosmological models.

    This class provides an interface for cosmological computations used in lensing
    such as comoving distance and critical surface density.

    Units
    -----
    Distance
        Mpc
    Mass
        solar mass

    Attributes
    ----------
    name: str
        Name of the cosmological model.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the Cosmology.

        Parameters
        ----------
        name: str
            Name of the cosmological model.
        """
        super().__init__(name)

    @abstractmethod
    def critical_density(self, z: Tensor, params: Optional["Packed"] = None) -> Tensor:
        """
        Compute the critical density at redshift z.

        Parameters
        ----------
        z: Tensor
            The redshifts.
        params: Packed, optional
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The critical density at each redshift.
        """
        ...

    @abstractmethod
    @unpack
    def comoving_distance(
        self, z: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the comoving distance to redshift z.

        Parameters
        ----------
        z: Tensor
            The redshifts.
        params: (Packed, optional0
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The comoving distance to each redshift.
        """
        ...

    @abstractmethod
    @unpack
    def transverse_comoving_distance(
        self, z: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the transverse comoving distance to redshift z (Mpc).

        Parameters
        ----------
        z: Tensor
            The redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The transverse comoving distance to each redshift in Mpc.
        """
        ...

    @unpack
    def comoving_distance_z1z2(
        self, z1: Tensor, z2: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the comoving distance between two redshifts.

        Parameters
        ----------
        z1: Tensor
            The starting redshifts.
        z2: Tensor
            The ending redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The comoving distance between each pair of redshifts.
        """
        return _comoving_distance_z1z2(self.comoving_distance, z1, z2, params=params)

    @unpack
    def transverse_comoving_distance_z1z2(
        self, z1: Tensor, z2: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the transverse comoving distance between two redshifts (Mpc).

        Parameters
        ----------
        z1: Tensor
            The starting redshifts.
        z2: Tensor
            The ending redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The transverse comoving distance between each pair of redshifts in Mpc.
        """
        return _transverse_comoving_distance_z1z2(
            self.transverse_comoving_distance, z1, z2, params=params
        )

    @unpack
    def angular_diameter_distance(
        self, z: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the angular diameter distance to redshift z.

        Parameters
        -----------
        z: Tensor
            The redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The angular diameter distance to each redshift.
        """
        return _angular_diameter_distance(
            self.comoving_distance, z, params=params, **kwargs
        )

    @unpack
    def angular_diameter_distance_z1z2(
        self, z1: Tensor, z2: Tensor, *args, params: Optional["Packed"] = None, **kwargs
    ) -> Tensor:
        """
        Compute the angular diameter distance between two redshifts.

        Parameters
        ----------
        z1: Tensor
            The starting redshifts.
        z2: Tensor
            The ending redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The angular diameter distance between each pair of redshifts.
        """
        return _angular_diameter_distance_z1z2(
            self.comoving_distance_z1z2, z1, z2, params=params, **kwargs
        )

    @unpack
    def time_delay_distance(
        self,
        z_l: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the time delay distance between lens and source planes.

        Parameters
        ----------
        z_l: Tensor
            The lens redshifts.
        z_s: Tensor
            The source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The time delay distance for each pair of lens and source redshifts.
        """
        return _time_delay_distance(
            self.angular_diameter_distance,
            self.angular_diameter_distance_z1z2,
            z_l,
            z_s,
            params=params,
        )

    @unpack
    def critical_surface_density(
        self,
        z_l: Tensor,
        z_s: Tensor,
        *args,
        params: Optional["Packed"] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute the critical surface density between lens and source planes.

        Parameters
        ----------
        z_l: Tensor
            The lens redshifts.
        z_s: Tensor
            The source redshifts.
        params: (Packed, optional)
            Dynamic parameter container for the computation.

        Returns
        -------
        Tensor
            The critical surface density for each pair of lens and source redshifts.
        """
        return _critical_surface_density(
            self.angular_diameter_distance,
            self.angular_diameter_distance_z1z2,
            z_l,
            z_s,
            params=params,
        )
