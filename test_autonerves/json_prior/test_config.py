

def test_lookup_cache_repeats_and_misses():
    """
    Lookups are memoised per instance — including misses, which the class-family
    probe in for_class_and_suffix_path performs repeatedly by design.
    """
    import pytest
    from autonerves.json_prior.config import JSONPriorConfig

    config = JSONPriorConfig({"module.Class": {"value": 1}})

    assert config(["pkg", "module", "Class"]) == {"value": 1}
    assert config(["pkg", "module", "Class"]) == {"value": 1}

    with pytest.raises(KeyError):
        config(["pkg", "module", "Missing"])
    with pytest.raises(KeyError):
        config(["pkg", "module", "Missing"])
