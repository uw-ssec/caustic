from math import pi
from io import StringIO

import lenstronomy.Util.param_util as param_util
import torch
from lenstronomy.LensModel.lens_model import LensModel
from utils import lens_test_helper

from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE
from caustics.utils import meshgrid
from caustics.sims import build_simulator

import pytest


@pytest.mark.parametrize("q", [0.5, 0.7, 0.9])
@pytest.mark.parametrize("phi", [pi / 3, -pi / 4, pi / 6])
@pytest.mark.parametrize("th_ein", [0.1, 1.0, 2.5])
def test_sie(sim_source, device, q, phi, th_ein):
    atol = 1e-5
    rtol = 1e-3

    if sim_source == "yaml":
        yaml_str = """\
        cosmology: &cosmology
            name: cosmo
            kind: FlatLambdaCDM
        lens: &lens
            name: sie
            kind: SIE
            init_kwargs:
                cosmology: *cosmology
        """
        with StringIO(yaml_str) as f:
            lens = build_simulator(f)
    else:
        # Models
        cosmology = FlatLambdaCDM(name="cosmo")
        lens = SIE(name="sie", cosmology=cosmology)
    lens_model_list = ["SIE"]
    lens_ls = LensModel(lens_model_list=lens_model_list)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([0.5, 0.912, -0.442, q, phi, th_ein])
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi, q=q)
    kwargs_ls = [
        {
            "theta_E": th_ein,
            "e1": e1,
            "e2": e2,
            "center_x": x[1].item(),
            "center_y": x[2].item(),
        }
    ]

    lens_test_helper(lens, lens_ls, z_s, x, kwargs_ls, rtol, atol, device=device)


def test_sie_time_delay():
    # Models
    cosmology = FlatLambdaCDM(name="cosmo")
    lens = SIE(name="sie", cosmology=cosmology)

    # Parameters
    z_s = torch.tensor(1.2)
    x = torch.tensor([0.5, 0.912, -0.442, 0.7, pi / 3, 1.4])

    n_pix = 10
    res = 0.05
    upsample_factor = 2
    thx, thy = meshgrid(
        res / upsample_factor,
        upsample_factor * n_pix,
        upsample_factor * n_pix,
        dtype=torch.float32,
    )

    assert torch.all(torch.isfinite(lens.time_delay(thx, thy, z_s, x)))
    assert torch.all(
        torch.isfinite(
            lens.time_delay(
                thx,
                thy,
                z_s,
                x,
                geometric_time_delay=True,
                shapiro_time_delay=False,
            )
        )
    )
    assert torch.all(
        torch.isfinite(
            lens.time_delay(
                thx,
                thy,
                z_s,
                x,
                geometric_time_delay=False,
                shapiro_time_delay=True,
            )
        )
    )
