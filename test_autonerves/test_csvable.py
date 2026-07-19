import pytest

from autonerves.csvable import list_from_csv, output_to_csv


def test_round_trip__uniform_dict_rows(tmp_path):
    rows = [
        {"a": "1", "b": "x"},
        {"a": "2", "b": "y"},
        {"a": "3", "b": "z"},
    ]
    path = tmp_path / "uniform.csv"

    output_to_csv(rows, path)
    loaded = list_from_csv(path)

    assert loaded == rows
    assert list(loaded[0].keys()) == ["a", "b"]


def test_round_trip__list_of_lists_with_explicit_headers(tmp_path):
    headers = ["a", "b", "c"]
    rows = [["1", "2", "3"], ["4", "5", "6"]]
    path = tmp_path / "seq.csv"

    output_to_csv(rows, path, headers=headers)
    loaded = list_from_csv(path)

    assert loaded == [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "4", "b": "5", "c": "6"},
    ]
    assert list(loaded[0].keys()) == headers


def test_flexible_headers__union_in_first_appearance_order(tmp_path):
    rows = [
        {"name": "s1", "y": "0.1", "x": "0.2"},
        {"name": "s1", "y": "0.3", "x": "0.4", "flux": "1.0"},
        {"name": "s2", "y": "0.5", "x": "0.6"},
    ]
    path = tmp_path / "flex.csv"

    output_to_csv(rows, path)
    loaded = list_from_csv(path)

    assert list(loaded[0].keys()) == ["name", "y", "x", "flux"]
    assert loaded[0]["flux"] == ""
    assert loaded[1]["flux"] == "1.0"
    assert loaded[2]["flux"] == ""


def test_explicit_headers__drops_extra_keys(tmp_path):
    rows = [
        {"a": "1", "b": "x", "ignored": "skip"},
        {"a": "2", "b": "y"},
    ]
    path = tmp_path / "drop.csv"

    output_to_csv(rows, path, headers=["a", "b"])
    loaded = list_from_csv(path)

    assert loaded == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def test_explicit_headers__missing_key_is_blank(tmp_path):
    rows = [{"a": "1", "b": "x"}, {"a": "2"}]
    path = tmp_path / "missing.csv"

    output_to_csv(rows, path, headers=["a", "b"])
    loaded = list_from_csv(path)

    assert loaded == [{"a": "1", "b": "x"}, {"a": "2", "b": ""}]


def test_empty_rows_with_headers__header_only_round_trips_to_empty(tmp_path):
    path = tmp_path / "empty.csv"

    output_to_csv([], path, headers=["a", "b"])
    loaded = list_from_csv(path)

    assert loaded == []
    with open(path) as f:
        assert f.read().splitlines() == ["a,b"]


def test_empty_rows_no_headers__writes_empty_file(tmp_path):
    path = tmp_path / "nothing.csv"

    output_to_csv([], path)
    loaded = list_from_csv(path)

    assert loaded == []
    assert path.read_text() == ""


def test_parent_directory_auto_created(tmp_path):
    path = tmp_path / "new_dir" / "nested" / "out.csv"

    output_to_csv([{"a": "1"}], path)

    assert path.exists()
    assert list_from_csv(path) == [{"a": "1"}]


def test_row_order_and_header_order_preserved(tmp_path):
    rows = [{"b": str(i), "a": str(i * 10)} for i in range(10)]
    path = tmp_path / "order.csv"

    output_to_csv(rows, path)
    loaded = list_from_csv(path)

    assert list(loaded[0].keys()) == ["b", "a"]
    assert [r["b"] for r in loaded] == [str(i) for i in range(10)]
    assert [r["a"] for r in loaded] == [str(i * 10) for i in range(10)]


def test_list_of_lists_without_headers_raises(tmp_path):
    with pytest.raises(ValueError, match="headers must be provided"):
        output_to_csv([[1, 2, 3]], tmp_path / "bad.csv")
