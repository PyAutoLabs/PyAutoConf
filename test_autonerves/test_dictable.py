import json

import numpy as np
import pytest
from pathlib import Path

from autonerves.dictable import to_dict, from_dict, register_parser


@pytest.fixture(name="array_dict")
def make_array_dict():
    return {"array": [1.0], "dtype": "float64", "type": "ndarray"}


@pytest.fixture(name="array")
def make_array():
    return np.array([1.0])


class ArrayImpl:
    def __init__(self, array):
        self.array = array

    @property
    def dtype(self):
        return self.array.dtype

    def tolist(self):
        return self.array.tolist()

    def __array__(self):
        return self.array


def test_array_impl(array):
    assert to_dict(ArrayImpl(array)) == to_dict(array)


def test_array_as_dict(array_dict, array):
    assert to_dict(array) == array_dict


def test_from_dict(array_dict, array):
    assert from_dict(array_dict) == array


@pytest.mark.parametrize(
    "array",
    [
        np.array([True]),
        np.array([[1.0]]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[1, 2], [3, 4]]),
    ],
)
def test_multiple(array):
    assert (from_dict(to_dict(array)) == array).all()


def test_as_json(array):
    assert from_dict(json.loads(json.dumps(to_dict(array)))) == array


def test_with_type_attribute():
    float_dict = {"class_path": "float", "type": "type"}
    assert to_dict(float) == float_dict
    assert from_dict(float_dict) is float


def test_register_parser():
    register_parser("test", lambda x: x["value"])
    assert from_dict({"type": "test", "value": 1}) == 1


def test_no_type():
    assert from_dict({"hi": "there"}) == {"hi": "there"}


def test_serialise_path():
    path = Path("/path/to/file.json")
    path_dict = to_dict(path)
    assert from_dict(path_dict) == path


class Parent:
    def __init__(self, parent_arg):
        self.parent_arg = parent_arg


class Child(Parent):
    def __init__(self, child_arg, **kwargs):
        super().__init__(**kwargs)
        self.child_arg = child_arg


def test_serialise_kwargs():
    child = Child(
        child_arg="child",
        parent_arg="parent",
    )
    child_dict = to_dict(child)
    assert child_dict == {
        "arguments": {
            "child_arg": "child",
            "parent_arg": "parent",
        },
        "class_path": "test_autonerves.test_dictable.Child",
        "type": "instance",
    }
    new_child = from_dict(child_dict)
    assert new_child.child_arg == "child"
    assert new_child.parent_arg == "parent"


class WithOptional:
    def __init__(self, arg: int = 1):
        self.arg = arg


def test_serialise_with_arg():
    assert to_dict(WithOptional()) == {
        "arguments": {"arg": 1},
        "class_path": "test_autonerves.test_dictable.WithOptional",
        "type": "instance",
    }


def test_serialise_without_arg():
    assert to_dict(WithOptional(), filter_args=("arg",)) == {
        "arguments": {},
        "class_path": "test_autonerves.test_dictable.WithOptional",
        "type": "instance",
    }


class C:
    pass


def test_tuple():
    assert from_dict(
        json.loads(
            json.dumps(
                to_dict(
                    (C, C),
                )
            )
        )
    ) == (C, C)


def function():
    return 1


def test_function():
    assert (
        from_dict(
            json.loads(
                json.dumps(
                    to_dict(
                        function,
                    )
                )
            )
        )()
        == 1
    )


def test_slice():
    s = slice(1, 2, 3)

    assert s == from_dict(to_dict(s))


def test_int_64():
    i = np.int64(1)
    result = from_dict(to_dict(i))

    assert result == 1
    assert isinstance(result, np.int64)


def test_int64_slice():
    s = slice(np.int64(1), np.int64(2), np.int64(3))

    assert s == from_dict(
        json.loads(
            json.dumps(
                to_dict(s),
            )
        )
    )


def test_compound_key():
    d = {(1, 2): 1}

    string = json.dumps(to_dict(d))
    assert d == from_dict(json.loads(string))


class Prior:
    def __eq__(self, other):
        return isinstance(other, Prior)

    def __hash__(self):
        return hash(Prior)


def test_prior_key():
    string = json.dumps(to_dict({Prior(): 1}))
    assert from_dict(json.loads(string)) == {Prior(): 1}
