"""Strict integer-validation utilities preventing silent/lossy coercions."""

from __future__ import annotations

import datetime
import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
_INT64_MIN_FLOAT = float(INT64_MIN)
_INT64_UPPER_BOUND_FLOAT = float(1 << 63)  # Exact power of 2; values >= this overflow int64

_BASE10_INT_RE = re.compile(r"[+-]?[0-9]+")
_BOOL_TOKENS = frozenset({"true", "false"})
_MISSING_TOKENS = frozenset({"nan", "+nan", "-nan", "na", "n/a", "none", "null"})
_INF_TOKENS = frozenset({"inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"})


def _check_bounds_array(
    values: np.ndarray,
    *,
    name: str,
    min_value: int | None,
    max_value: int | None,
) -> None:
    """Verify every element of ``values`` satisfies optional inclusive bounds."""
    if len(values) == 0:
        return

    below = (values < min_value) if min_value is not None else np.zeros(len(values), dtype=bool)
    above = (values > max_value) if max_value is not None else np.zeros(len(values), dtype=bool)
    out_of_range = below | above
    if np.any(out_of_range):
        bad_values = np.unique(values[out_of_range])[:5].tolist()
        if min_value is not None and max_value is not None:
            bounds = f"[{min_value}, {max_value}]"
        elif min_value is not None:
            bounds = f">= {min_value}"
        else:
            bounds = f"<= {max_value}"
        raise ValueError(
            f"Invalid {name} values: expected integers in {bounds}, "
            f"found out-of-range value(s): {bad_values}."
        )


def _parse_strict_int_string(
    token: str,
    *,
    name: str,
    allow_integral_floats: bool,
) -> int:
    """Parse a string token into an exact Python int without using float64."""
    stripped = token.strip()
    if stripped == "":
        raise ValueError(f"Invalid {name} values: found missing/empty value.")

    lowered = stripped.lower()
    if lowered in _BOOL_TOKENS:
        raise ValueError(
            f"Invalid {name} values: boolean value {token!r} is not permitted."
        )
    if lowered in _MISSING_TOKENS:
        raise ValueError(
            f"Invalid {name} values: missing value {token!r} detected."
        )
    if lowered in _INF_TOKENS:
        raise ValueError(
            f"Invalid {name} values: non-finite value {token!r} detected."
        )

    if _BASE10_INT_RE.fullmatch(stripped):
        parsed = int(stripped)
        if parsed < INT64_MIN or parsed > INT64_MAX:
            raise ValueError(
                f"Invalid {name} values: integer {parsed} is out-of-range for int64."
            )
        return parsed

    if "_" in stripped:
        raise ValueError(
            f"Invalid {name} values: non-numeric or unparseable value {token!r}."
        )

    try:
        dec = Decimal(stripped)
    except InvalidOperation as exc:
        raise ValueError(
            f"Invalid {name} values: non-numeric or unparseable value {token!r}."
        ) from exc

    if dec.is_nan():
        raise ValueError(
            f"Invalid {name} values: missing/non-finite value {token!r} detected."
        )
    if not dec.is_finite():
        raise ValueError(
            f"Invalid {name} values: non-finite value {token!r} detected."
        )

    sign, digits, exponent = dec.as_tuple()
    if not isinstance(exponent, int):
        raise ValueError(
            f"Invalid {name} values: non-finite value {token!r} detected."
        )

    if exponent < 0:
        frac_start = len(digits) + exponent
        frac_digits = digits[max(frac_start, 0) :]
        if any(d != 0 for d in frac_digits):
            raise ValueError(
                f"Invalid {name} values: fractional values are not permitted; found {token!r}."
            )
        if not allow_integral_floats:
            raise ValueError(
                f"Invalid {name} values: non-integer decimal value {token!r} is not permitted."
            )
        int_digits = digits[:frac_start] if frac_start > 0 else (0,)
        magnitude = int("".join(str(d) for d in int_digits))
        parsed = -magnitude if sign else magnitude
    else:
        if not allow_integral_floats:
            raise ValueError(
                f"Invalid {name} values: scientific/decimal notation {token!r} is not permitted."
            )
        # Guard against huge exponents (e.g. 1e100000) before exact expansion.
        nz_digits = digits
        first_nz = next((i for i, d in enumerate(nz_digits) if d != 0), None)
        if first_nz is None:
            parsed = 0
        else:
            sig_len = len(nz_digits) - first_nz
            if sig_len + exponent > 20:
                raise ValueError(
                    f"Invalid {name} values: value {token!r} is out-of-range for int64."
                )
            magnitude = int("".join(str(d) for d in nz_digits[first_nz:])) * (10**exponent)
            parsed = -magnitude if sign else magnitude

    if parsed < INT64_MIN or parsed > INT64_MAX:
        raise ValueError(
            f"Invalid {name} values: integer {parsed} is out-of-range for int64."
        )
    return parsed


def _parse_strict_int_scalar(
    item: Any,
    *,
    name: str,
    allow_integral_floats: bool,
) -> int:
    """Validate and convert one scalar element to an exact int64-compatible Python int."""
    if item is None or item is pd.NA or item is pd.NaT or item is np.ma.masked:
        raise ValueError(f"Invalid {name} values: found missing/masked value.")

    if isinstance(item, (bool, np.bool_)):
        raise ValueError(
            f"Invalid {name} values: boolean values are not permitted; found {item!r}."
        )

    if isinstance(item, (complex, np.complexfloating)):
        raise ValueError(
            f"Invalid {name} values: complex values are not permitted; found {item!r}."
        )

    if isinstance(
        item,
        (
            np.datetime64,
            np.timedelta64,
            pd.Timestamp,
            pd.Timedelta,
            datetime.date,
            datetime.datetime,
            datetime.timedelta,
        ),
    ):
        raise ValueError(
            f"Invalid {name} values: datetime/timedelta values are not permitted; found {item!r}."
        )

    if isinstance(item, (int, np.integer)):
        parsed = int(item)
        if parsed < INT64_MIN or parsed > INT64_MAX:
            raise ValueError(
                f"Invalid {name} values: integer {parsed} is out-of-range for int64."
            )
        return parsed

    if isinstance(item, (float, np.floating)):
        fval = float(item)
        if math.isnan(fval):
            raise ValueError(f"Invalid {name} values: found missing/non-finite NaN value.")
        if not math.isfinite(fval):
            raise ValueError(f"Invalid {name} values: found non-finite value {item!r}.")
        if math.floor(fval) != fval:
            raise ValueError(
                f"Invalid {name} values: fractional values are not permitted; found {item!r}."
            )
        if not allow_integral_floats:
            raise ValueError(
                f"Invalid {name} values: floating-point value {item!r} is not permitted."
            )
        if fval < _INT64_MIN_FLOAT or fval >= _INT64_UPPER_BOUND_FLOAT:
            raise ValueError(
                f"Invalid {name} values: value {item!r} is out-of-range for int64."
            )
        return int(fval)

    if isinstance(item, Decimal):
        return _parse_strict_int_string(
            str(item),
            name=name,
            allow_integral_floats=allow_integral_floats,
        )

    if isinstance(item, str):
        return _parse_strict_int_string(
            item,
            name=name,
            allow_integral_floats=allow_integral_floats,
        )

    if isinstance(item, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        raise ValueError(f"Invalid {name} values: expected a 1D sequence, got nested sequence.")

    raise ValueError(
        f"Invalid {name} values: unsupported or non-numeric value {item!r} "
        f"(type {type(item).__name__})."
    )


def _validate_float_array(
    arr: np.ndarray,
    *,
    name: str,
    min_value: int | None,
    max_value: int | None,
    allow_integral_floats: bool,
) -> np.ndarray:
    """Validate a 1D floating-point NumPy array before casting to int64."""
    if len(arr) == 0:
        return np.empty(0, dtype=np.int64)

    nan_mask = np.isnan(arr)
    if np.any(nan_mask):
        raise ValueError(
            f"Invalid {name} values: found {int(np.sum(nan_mask))} missing/non-finite value(s)."
        )

    non_finite = ~np.isfinite(arr)
    if np.any(non_finite):
        raise ValueError(
            f"Invalid {name} values: found {int(np.sum(non_finite))} non-finite value(s)."
        )

    fractional = np.floor(arr) != arr
    if np.any(fractional):
        bad_values = np.unique(arr[fractional])[:5].tolist()
        raise ValueError(
            f"Invalid {name} values: fractional values are not permitted; found: {bad_values}."
        )

    if not allow_integral_floats:
        raise ValueError(
            f"Invalid {name} values: floating-point values are not permitted; "
            "expected exact integers."
        )

    # Promote narrower float dtypes (float16, float32) losslessly to float64
    # before comparing against int64 bounds to avoid NEP 50 scalar cast overflow warnings.
    arr_check = arr.astype(np.float64, copy=False) if arr.dtype.itemsize < 8 else arr

    if np.any((arr_check < _INT64_MIN_FLOAT) | (arr_check >= _INT64_UPPER_BOUND_FLOAT)):
        raise ValueError(
            f"Invalid {name} values: found out-of-range value(s) outside int64 bounds."
        )

    out = np.array(arr_check, dtype=np.int64, copy=True, order="C")
    _check_bounds_array(out, name=name, min_value=min_value, max_value=max_value)
    return out


def _validate_string_series(
    series: pd.Series,
    *,
    name: str,
    min_value: int | None,
    max_value: int | None,
    allow_integral_floats: bool,
) -> np.ndarray:
    """Validate a Pandas string Series using exact base-10 / Decimal parsing (no float64)."""
    if len(series) == 0:
        return np.empty(0, dtype=np.int64)

    if series.isna().any():
        missing_count = int(series.isna().sum())
        raise ValueError(
            f"Invalid {name} values: found {missing_count} missing value(s)."
        )

    stripped = series.str.strip()
    int_mask = stripped.str.fullmatch(_BASE10_INT_RE)

    if bool(int_mask.all()):
        max_len = int(stripped.str.len().max())
        # Signed int64 fits every base-10 integer with <= 18 characters (including optional sign).
        if max_len <= 18:
            out = np.array(
                stripped.astype(np.int64).to_numpy(copy=True),
                dtype=np.int64,
                copy=True,
                order="C",
            )
        else:
            parsed_ints: list[int] = []
            for tok in stripped.tolist():
                val = int(tok)
                if val < INT64_MIN or val > INT64_MAX:
                    raise ValueError(
                        f"Invalid {name} values: integer {val} is out-of-range for int64."
                    )
                parsed_ints.append(val)
            out = np.array(parsed_ints, dtype=np.int64, copy=True, order="C")
    else:
        parsed_ints = [
            _parse_strict_int_string(
                tok,
                name=name,
                allow_integral_floats=allow_integral_floats,
            )
            for tok in series.tolist()
        ]
        out = np.array(parsed_ints, dtype=np.int64, copy=True, order="C")

    _check_bounds_array(out, name=name, min_value=min_value, max_value=max_value)
    return out


def _validate_numpy_array(
    arr: np.ndarray,
    *,
    name: str,
    min_value: int | None,
    max_value: int | None,
    allow_integral_floats: bool,
) -> np.ndarray:
    """Validate a NumPy array without silent lossy coercions."""
    if arr.ndim != 1:
        raise ValueError(
            f"Invalid {name} values: expected a 1D sequence, got shape {arr.shape}."
        )

    if np.ma.isMaskedArray(arr):
        if np.ma.is_masked(arr):
            masked_count = int(np.sum(np.ma.getmaskarray(arr)))
            raise ValueError(
                f"Invalid {name} values: found {masked_count} masked/missing value(s)."
            )
        arr = np.asarray(arr.data)

    dtype = arr.dtype
    if np.issubdtype(dtype, np.bool_):
        raise ValueError(f"Invalid {name} values: boolean values are not permitted.")
    if np.issubdtype(dtype, np.complexfloating):
        raise ValueError(f"Invalid {name} values: complex values are not permitted.")
    if np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64):
        raise ValueError(f"Invalid {name} values: datetime/timedelta values are not permitted.")

    if len(arr) == 0:
        return np.empty(0, dtype=np.int64)

    if np.issubdtype(dtype, np.integer):
        if np.issubdtype(dtype, np.unsignedinteger) and np.any(arr > np.uint64(INT64_MAX)):
            raise ValueError(
                f"Invalid {name} values: unsigned integer exceeds int64 maximum."
            )
        out = np.array(arr, dtype=np.int64, copy=True, order="C")
        _check_bounds_array(out, name=name, min_value=min_value, max_value=max_value)
        return out

    if np.issubdtype(dtype, np.floating):
        return _validate_float_array(
            arr,
            name=name,
            min_value=min_value,
            max_value=max_value,
            allow_integral_floats=allow_integral_floats,
        )

    if np.issubdtype(dtype, np.str_):
        return _validate_string_series(
            pd.Series(arr, dtype="string"),
            name=name,
            min_value=min_value,
            max_value=max_value,
            allow_integral_floats=allow_integral_floats,
        )

    # Object or other structured dtypes: validate each scalar strictly.
    parsed = [
        _parse_strict_int_scalar(item, name=name, allow_integral_floats=allow_integral_floats)
        for item in arr.tolist()
    ]
    out = np.array(parsed, dtype=np.int64, copy=True, order="C")
    _check_bounds_array(out, name=name, min_value=min_value, max_value=max_value)
    return out


def validate_strict_int64(
    values: Any,
    *,
    name: str,
    min_value: int | None = None,
    max_value: int | None = None,
    allow_integral_floats: bool = True,
) -> np.ndarray:
    """Strictly validate a 1D sequence and return an owned, writable ``int64`` array.

    Never relies on lossy NumPy/Pandas coercions (such as ``np.asarray`` on mixed
    ``[True, 2]`` lists, ``pd.to_numeric`` on ``datetime64``/``complex``, or
    ``float64`` parsing of high-precision integer/decimal strings).
    """
    if isinstance(values, (str, bytes)):
        raise ValueError(
            f"Invalid {name} values: expected a 1D sequence, got scalar {type(values).__name__}."
        )

    if isinstance(values, np.ndarray):
        return _validate_numpy_array(
            values,
            name=name,
            min_value=min_value,
            max_value=max_value,
            allow_integral_floats=allow_integral_floats,
        )

    if isinstance(values, (pd.Series, pd.Index)):
        series = values.to_series(index=None) if isinstance(values, pd.Index) else values
        dtype = series.dtype

        if isinstance(dtype, pd.BooleanDtype) or pd.api.types.is_bool_dtype(dtype):
            raise ValueError(f"Invalid {name} values: boolean values are not permitted.")
        if pd.api.types.is_complex_dtype(dtype):
            raise ValueError(f"Invalid {name} values: complex values are not permitted.")
        if (
            pd.api.types.is_datetime64_any_dtype(dtype)
            or pd.api.types.is_timedelta64_dtype(dtype)
            or isinstance(dtype, pd.PeriodDtype)
        ):
            raise ValueError(
                f"Invalid {name} values: datetime/timedelta values are not permitted."
            )

        if len(series) == 0:
            return np.empty(0, dtype=np.int64)

        if series.isna().any():
            missing_count = int(series.isna().sum())
            raise ValueError(
                f"Invalid {name} values: found {missing_count} missing value(s)."
            )

        if pd.api.types.is_integer_dtype(dtype):
            arr = series.to_numpy(copy=True)
            return _validate_numpy_array(
                arr,
                name=name,
                min_value=min_value,
                max_value=max_value,
                allow_integral_floats=allow_integral_floats,
            )

        if pd.api.types.is_float_dtype(dtype):
            arr = series.to_numpy(dtype=np.float64, copy=True)
            return _validate_float_array(
                arr,
                name=name,
                min_value=min_value,
                max_value=max_value,
                allow_integral_floats=allow_integral_floats,
            )

        if isinstance(dtype, pd.StringDtype):
            return _validate_string_series(
                series,
                name=name,
                min_value=min_value,
                max_value=max_value,
                allow_integral_floats=allow_integral_floats,
            )

        # Object / mixed extension Series: inspect every element directly.
        raw_items = series.tolist()
        if all(type(v) is str for v in raw_items):
            return _validate_string_series(
                series.astype("string"),
                name=name,
                min_value=min_value,
                max_value=max_value,
                allow_integral_floats=allow_integral_floats,
            )

        parsed = [
            _parse_strict_int_scalar(item, name=name, allow_integral_floats=allow_integral_floats)
            for item in raw_items
        ]
        out = np.array(parsed, dtype=np.int64, copy=True, order="C")
        _check_bounds_array(out, name=name, min_value=min_value, max_value=max_value)
        return out

    # Generic Python iterable (list, tuple, range, generator): inspect elements BEFORE NumPy.
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(
            f"Invalid {name} values: expected a 1D sequence, got {type(values).__name__}."
        ) from exc

    if len(items) == 0:
        return np.empty(0, dtype=np.int64)

    parsed = [
        _parse_strict_int_scalar(item, name=name, allow_integral_floats=allow_integral_floats)
        for item in items
    ]
    out = np.array(parsed, dtype=np.int64, copy=True, order="C")
    _check_bounds_array(out, name=name, min_value=min_value, max_value=max_value)
    return out
