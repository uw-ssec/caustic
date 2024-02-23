# mypy: disable-error-code="import-untyped"
from ..sims.simulator import Simulator

from .class_models import Config
from ..io import from_file


def build_simulator(config: "str | Config") -> Simulator:
    """
    Build a simulator from the configuration
    """
    if isinstance(config, str):
        import yaml

        # Load the yaml config
        yaml_bytes = from_file(config)
        config_json = yaml.safe_load(yaml_bytes)
        # Create config model
        config = Config(**config_json)

    # Get the simulator
    sim = config.simulator.model_obj()

    # Load state if available
    if config.state:
        sim.load_state_dict(config.state.load.path)

    return sim
