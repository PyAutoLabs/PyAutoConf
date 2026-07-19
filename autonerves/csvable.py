"""
Generic CSV reader/writer helpers for the PyAuto ecosystem.

Sits alongside :mod:`autonerves.dictable` (JSON) and :mod:`autonerves.fitsable`
(FITS) as the third text-format I/O surface. The functions here are schema
agnostic — callers layer their own column conventions on top (see e.g.
``autolens.point.dataset`` for the PointDataset schema layer).

Only the standard-library ``csv`` module is used; there is no pandas
dependency.
"""
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import csv


Row = Union[dict, Sequence]


def output_to_csv(
    rows: Iterable[Row],
    file_path: Union[str, Path],
    headers: Optional[List[str]] = None,
):
    """
    Write ``rows`` to ``file_path`` as a CSV.

    Parameters
    ----------
    rows
        Either a list of dicts (``{column: value}``) or a list of sequences.
    file_path
        Destination path. Parent directories are created if missing.
    headers
        Optional explicit column list.

        - For dict rows with ``headers=None``: the header row is the union
          of keys across all rows in first-appearance order — a column is
          written if *any* row populates it, and rows that omit the key
          get a blank cell.
        - For dict rows with explicit ``headers``: the given columns are
          used verbatim; extra keys in any row are dropped silently; missing
          keys produce blanks.
        - For sequence rows: ``headers`` is required.
    """
    rows = list(rows)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    is_dict_rows = bool(rows) and isinstance(rows[0], dict)

    if not rows:
        with open(file_path, "w", newline="") as f:
            if headers:
                csv.writer(f).writerow(headers)
        return

    if is_dict_rows:
        if headers is None:
            headers = []
            seen = set()
            for row in rows:
                for key in row:
                    if key not in seen:
                        seen.add(key)
                        headers.append(key)

        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=headers, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(rows)
        return

    if headers is None:
        raise ValueError(
            "output_to_csv: headers must be provided when rows are sequences "
            "(not dicts); sequence rows carry no column names of their own."
        )

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def list_from_csv(file_path: Union[str, Path]) -> List[dict]:
    """
    Read a CSV and return its rows as an ordered list of dicts.

    Row order is preserved.  Within each row, keys are ordered to match the
    header line (Python dicts are insertion-ordered and :class:`csv.DictReader`
    inserts fields in ``fieldnames`` order), so callers that need the header
    list can recover it with ``list(rows[0].keys())`` when at least one row
    is present.

    An empty CSV (no header line) and a header-only CSV (header line but no
    data rows) both return an empty list.
    """
    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
