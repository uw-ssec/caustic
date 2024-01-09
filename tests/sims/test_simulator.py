import pytest

from caustics.sims.simulator import Simulator
from caustics.lenses import EPL
from caustics.light import Sersic
from caustics.cosmology import FlatLambdaCDM
from caustics.sims.state_dict import StateDict


@pytest.fixture
def simple_common_sim():
    class Sim(Simulator):
        def __init__(self):
            super().__init__()
            self.cosmo = FlatLambdaCDM(h0=None)
            self.epl = EPL(self.cosmo)
            self.sersic = Sersic()
            self.add_param("z_s", 1.0)

    sim = Sim()
    yield sim
    del sim


@pytest.fixture
def state_dict(simple_common_sim):
    return simple_common_sim.state_dict()


@pytest.fixture
def expected_tensors(simple_common_sim):
    static_params = simple_common_sim.params["static"].flatten()
    return {k: v.value for k, v in static_params.items()}


class TestSimulator:
    def test_state_dict(self, state_dict, expected_tensors):
        # Check state_dict type and default keys
        assert isinstance(state_dict, StateDict)

        # Trying to modify state_dict should raise TypeError
        with pytest.raises(TypeError):
            state_dict["params"] = -1

        # Check _metadata keys
        assert "software_version" in state_dict._metadata
        assert "created_time" in state_dict._metadata

        # Check params
        assert dict(state_dict) == expected_tensors
