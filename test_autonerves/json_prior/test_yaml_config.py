from .source_code.subdirectory.subconfig import SubClass


class YAMLClass:
    def __init__(self, variable: float):
        self.variable = variable


def test_load_yaml_config(config):
    assert config.prior_config.for_class_and_suffix_path(YAMLClass, ["variable"]) == {
        "lower_limit": 0.0,
        "type": "Uniform",
        "upper_limit": 3.0,
    }


def test_embedded_path(config):
    path_value_map = config.prior_config.prior_configs[0].path_value_map
    assert "subdirectory.subconfig.SubClass.variable.type" in path_value_map


def test_subdirectory(config):
    assert config.prior_config.for_class_and_suffix_path(SubClass, ["variable"]) == {
        "lower_limit": 0.0,
        "type": "Uniform",
        "upper_limit": 3.0,
    }
