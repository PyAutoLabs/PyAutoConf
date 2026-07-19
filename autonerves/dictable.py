import inspect
import json
import logging
import os

import numpy as np
from pathlib import Path
from typing import Union, Callable, Set, Tuple

from autonerves.class_path import get_class_path, get_class

logger = logging.getLogger(__name__)

np_type_map = {
    "bool": "bool_",
}


def nd_array_as_dict(obj: np.ndarray) -> dict:
    """
    Converts a numpy array to a dictionary representation.
    """
    np_type = str(obj.dtype)
    return {
        "type": "ndarray",
        "array": obj.tolist(),
        "dtype": np_type_map.get(np_type, np_type),
    }


def nd_array_from_dict(nd_array_dict: dict, **_) -> np.ndarray:
    """
    Converts a dictionary representation back to a numpy array.
    """
    return np.array(nd_array_dict["array"], dtype=getattr(np, nd_array_dict["dtype"]))


def is_array(obj) -> bool:
    """
    True if the object is a numpy array or an ArrayImpl (i.e. from JAX)
    """
    if isinstance(obj, np.ndarray):
        return True
    try:
        return obj.__class__.__name__ == "ArrayImpl"
    except AttributeError:
        return False


def compound_key_dict(obj):
    """
    Converts a dictionary with compound keys to a dictionary with a single key.
    """
    return {
        "type": "compound_dict",
        "arguments": [
            {
                "key": to_dict(key),
                "value": to_dict(value),
            }
            for key, value in obj.items()
        ],
    }


def to_dict(obj, filter_args: Tuple[str, ...] = ()) -> dict:
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except TypeError as e:
            logger.debug(e)

    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj

    if isinstance(obj, slice):
        return {
            "type": "slice",
            "start": to_dict(obj.start),
            "stop": to_dict(obj.stop),
            "step": to_dict(obj.step),
        }

    if isinstance(obj, np.number):
        return {
            "type": "np.number",
            "dtype": str(obj.dtype),
            "value": obj.item(),
        }

    if inspect.isfunction(obj):
        return {
            "type": "function",
            "class_path": obj.__module__ + "." + obj.__qualname__,
        }

    if is_array(obj):
        try:
            return nd_array_as_dict(obj)
        except Exception as e:
            logger.info(e)

    if isinstance(obj, Path):
        return {
            "type": "path",
            "path": str(obj),
        }

    if inspect.isclass(obj):
        return {
            "type": "type",
            "class_path": get_class_path(obj),
        }

    if isinstance(obj, list):
        return {"type": "list", "values": list(map(to_dict, obj))}
    if isinstance(obj, tuple):
        return {"type": "tuple", "values": list(map(to_dict, obj))}
    if isinstance(obj, dict):
        if any(
            not (isinstance(key, (str, int, float, bool)) or key is None)
            for key in obj.keys()
        ):
            return compound_key_dict(obj)

        return {
            "type": "dict",
            "arguments": {
                key: to_dict(value)
                for key, value in obj.items()
                if key not in filter_args
            },
        }

    if obj.__class__.__name__ == "method":
        return to_dict(obj())
    if obj.__class__.__module__ == "builtins":
        return obj

    if inspect.isclass(type(obj)):
        return instance_as_dict(obj, filter_args=filter_args)

    return obj


def get_arguments(obj) -> Set[str]:
    """
    Get the arguments of a class. This is done by inspecting the constructor.

    If the constructor has a **kwargs parameter, the arguments of the base classes are also included.

    Parameters
    ----------
    obj
        The class to get the arguments of.

    Returns
    -------
    A set of the arguments of the class.
    """
    args_spec = inspect.getfullargspec(obj.__init__)
    args = set(args_spec.args[1:])
    if args_spec.varkw:
        for base in obj.__bases__:
            if base is object:
                continue
            args |= get_arguments(base)
    return args


def instance_as_dict(obj, filter_args: Tuple[str, ...] = ()):
    """
    Convert an instance of a class to a dictionary representation.

    Serialises any children of the object which are given as constructor arguments
    or included in the __identifier_fields__ attribute.

    Sets any fields in the __nullify_fields__ attribute to None.

    Parameters
    ----------
    obj
        The instance of the class to be converted to a dictionary representation.
    filter_args
        A tuple of arguments to exclude from the dictionary representation.

    Returns
    -------
    A dictionary representation of the instance.
    """
    arguments = get_arguments(type(obj))
    try:
        arguments |= set(obj.__identifier_fields__)
    except (AttributeError, TypeError):
        pass

    argument_dict = {
        arg: getattr(obj, arg)
        for arg in arguments
        if arg not in filter_args
        if hasattr(obj, arg)
        and not inspect.ismethod(
            getattr(obj, arg),
        )
    }
    try:
        for field in obj.__nullify_fields__:
            argument_dict[field] = None
    except (AttributeError, TypeError):
        pass

    try:
        for field in obj.__exclude_fields__:
            try:
                argument_dict.pop(field)
            except KeyError:
                logger.debug(f"Field {field} not found in object")
    except (AttributeError, TypeError):
        pass

    return {
        "type": "instance",
        "class_path": get_class_path(obj.__class__),
        "arguments": {key: to_dict(value) for key, value in argument_dict.items()},
    }


__parsers = {
    "ndarray": nd_array_from_dict,
}


def register_parser(type_: str, parser: Callable[[dict], object]):
    """
    Register a parser for a given type.

    This parser will be used to instantiate objects of the given type from a
    dictionary representation.

    Parameters
    ----------
    type_
        The type of the object to be parsed. This is a string uniquely
        identifying the type.
    parser
        A function which takes a dictionary representation of an object and
        returns an instance of the object.
    """
    __parsers[type_] = parser


def from_dict(dictionary, **kwargs):
    """
    Instantiate an instance of a class from its dictionary representation.

    Parameters
    ----------
    dictionary
        An object which may be a dictionary representation of an object.

        This may contain the following keys:
        type: str
            The type of the object. This may be a built-in type, a numpy array,
            a list, a dictionary, a class, or an instance of a class.

            If a parser has been registered for the given type that parser will
            be used to instantiate the object.
        class_path: str
            The path to the class of the object. This is used to instantiate
            the object if it is not a built-in type.
        arguments: dict
            A dictionary of arguments to pass to the class constructor.

    Returns
    -------
    An object that was represented by the dictionary.
    """
    if isinstance(dictionary, (int, float, str, bool, type(None))):
        return dictionary

    if isinstance(dictionary, list):
        return list(map(from_dict, dictionary))

    if isinstance(dictionary, tuple):
        return tuple(map(from_dict, dictionary))

    try:
        type_ = dictionary["type"]
    except KeyError:
        logger.debug("No type field in dictionary")
        return dictionary
    except TypeError as e:
        logger.debug(e)
        return None

    if type_ == "path":
        return Path(dictionary["path"])

    if type_ == "slice":
        return slice(
            from_dict(dictionary["start"]),
            from_dict(dictionary["stop"]),
            from_dict(dictionary["step"]),
        )

    if type_ == "np.number":
        return getattr(
            np,
            dictionary["dtype"],
        )(dictionary["value"])

    if type_ == "function":
        return get_class(dictionary["class_path"])

    if type_ in __parsers:
        return __parsers[type_](dictionary, **kwargs)

    if type_ == "list":
        return list(map(from_dict, dictionary["values"]))
    if type_ == "tuple":
        return tuple(map(from_dict, dictionary["values"]))
    if type_ == "dict":
        return {
            key: from_dict(value, **kwargs)
            for key, value in dictionary["arguments"].items()
        }
    if type_ == "compound_dict":
        return {
            from_dict(item["key"], **kwargs): from_dict(item["value"], **kwargs)
            for item in dictionary["arguments"]
        }

    if type_ == "type":
        return get_class(dictionary["class_path"])

    cls = get_class(dictionary["class_path"])

    if cls is np.ndarray:
        return nd_array_from_dict(dictionary)
    if hasattr(cls, "from_dict"):
        return cls.from_dict(dictionary, **kwargs)

    # noinspection PyArgumentList
    return cls(
        **{
            name: from_dict(value, **kwargs)
            for name, value in dictionary["arguments"].items()
        }
    )


def from_json(file_path: str):
    """
    Load the dictable object to a .json file, whereby all attributes are converted from the .json file's dictionary
    representation to create the instance of the object

    A json file of the instance can be created from the .json file via the `output_to_json` method.

    Parameters
    ----------
    file_path
        The path to the .json file that the dictionary representation of the object is loaded from.
    """
    with open(file_path, "r+") as f:
        cls_dict = json.load(f)

    return from_dict(cls_dict)


def output_to_json(obj, file_path: Union[Path, str]):
    """
    Output the dictable object to a .json file, whereby all attributes are converted to a dictionary representation
    first.

    An instance of the object can be created from the .json file via the `from_json` method.

    Parameters
    ----------
    file_path
        The path to the .json file that the dictionary representation of the object is written too.
    """
    file_path = Path(file_path)
    file_dir = Path(*file_path.parts[:-1])
    file_dir.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w+") as f:
        json.dump(to_dict(obj), f, indent=4)
