import pytest

from caustics.sims.state_dict import StateDict


@pytest.fixture
def state_dict(simple_common_sim):
    return simple_common_sim.state_dict()


@pytest.fixture
def expected_tensors(simple_common_sim, sim_utils):
    return sim_utils.extract_tensors(simple_common_sim.params)


class TestSimulator:
    def test_state_dict(self, state_dict, expected_tensors, sim_utils):
        # Check state_dict type and default keys
        assert isinstance(state_dict, StateDict)

        # Trying to modify state_dict should raise TypeError
        with pytest.raises(TypeError):
            state_dict["params"] = -1

        # Check _metadata keys
        assert "software_version" in state_dict._metadata
        assert "created_time" in state_dict._metadata

        # Check params
        assert sim_utils.isEquals(dict(state_dict), expected_tensors)
