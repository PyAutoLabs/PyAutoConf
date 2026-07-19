from autonerves.tools.decorators import cached_property


class MockClass:
    def __init__(self, value):
        self._value = value

    @cached_property
    def value(self):
        return self._value


def test__profile_decorator_times_decorated_function(files_directory):
    cls = MockClass(value=1.0)
    cls.value

    assert cls.__dict__["value"] == 1.0
