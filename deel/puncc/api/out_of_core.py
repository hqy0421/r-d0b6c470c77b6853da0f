# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Out-of-core calibration artifacts.

This module implements a disk-backed calibration artifact for split conformal
prediction when the calibration data is spread over one or more Parquet shards
whose nonconformity scores cannot be held in memory.

The artifact is built by streaming every shard in batches, predicting with an
already trained predictor, and writing the resulting *scalar* nonconformity
scores (absolute residuals by default) to an externally sorted file. Duplicated
calibration samples are counted as many times as they appear: no de-duplication
is performed.

The stored scores jointly define the exact finite-sample quantile used by
:class:`deel.puncc.api.calibration.BaseCalibrator`, including the historical
convention of appending a single ``+inf`` score (Tibshirani's lemma) and the
``1/(n + 1) <= alpha < 1`` validity domain. No per-shard quantile is averaged
and no approximate quantile sketch is used: the requested order statistic is
fetched directly from the sorted file.

Once built, the artifact can be reopened without rescanning the source shards
and without loading the ``n`` scores in memory, and it exposes the same
calibration interface (:meth:`compute_quantile`, :meth:`calibrate`) as
:class:`~deel.puncc.api.calibration.BaseCalibrator` so that intervals can be
produced for new predictions.

Parquet shards are read with ``pyarrow``, which is an optional dependency::

    pip install "puncc[parquet]"

Example::

    from deel.puncc.api.out_of_core import build_calibration_artifact

    artifact = build_calibration_artifact(
        path="calibration.puncc",
        shards=["calib-000.parquet", "calib-001.parquet"],
        features=["f0", "f1", "f2"],
        target="y",
        predictor=my_trained_predictor,
        batch_size=4096,
    )
    alpha = 0.1
    y_lower, y_upper = artifact.calibrate(alpha=alpha, y_pred=y_pred_test)
    artifact.close()

    # Later, after the source shards are gone:
    from deel.puncc.api.out_of_core import open_calibration_artifact

    artifact = open_calibration_artifact("calibration.puncc")
    y_lower, y_upper = artifact.calibrate(alpha=alpha, y_pred=y_pred_test)
"""

# pylint: disable=C0115,C0116,C0415,R0912,R0913,R1732
from __future__ import annotations

import glob
import hashlib
import heapq
import importlib
import json
import logging
import os
import pickle
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

from deel.puncc.api import nonconformity_scores
from deel.puncc.api.backend import get_backend
from deel.puncc.api.calibration import BaseCalibrator
from deel.puncc.api.corrections import bonferroni
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.api.prediction_sets import constant_interval
from deel.puncc.api.utils import alpha_calib_check

logger = logging.getLogger(__name__)

FORMAT_NAME = "puncc-out-of-core-calibration"
FORMAT_VERSION = 1

METADATA_FILE = "metadata.json"
SCORES_FILE = "scores.f8"
MODEL_FILE = "model.pkl"
COMPLETE_MARKER = "_COMPLETE"

SCORES_DTYPE = np.dtype("<f8")
BYTES_PER_SCORE = 8

DEFAULT_BATCH_SIZE = 1024
DEFAULT_SORT_BUFFER_SCORES = 1 << 20
DEFAULT_MERGE_CHUNK_SCORES = 1 << 16
DEFAULT_MAX_FAN_IN = 32

_READ_CHUNK_BYTES = 1 << 20


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Errors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class CalibrationArtifactError(RuntimeError):
    """Base error raised by out-of-core calibration artifacts."""


class ArtifactNotReadyError(CalibrationArtifactError):
    """Raised when an artifact is missing, incomplete or corrupted."""


class ArtifactFormatError(CalibrationArtifactError):
    """Raised when an artifact does not match the expected format."""


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Helpers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_list(value: Any, name: str) -> List[Any]:
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, np.ndarray):
        return list(value.tolist())
    if isinstance(value, Sequence):
        return list(value)
    raise TypeError(
        f"`{name}` must be a sequence of names or a single name, "
        f"received {type(value)!r}."
    )


def _normalize_features(features: Any) -> List[str]:
    features = _as_list(features, "features")
    features = [str(f) for f in features]
    if len(features) == 0:
        raise ValueError("`features` must contain at least one column name.")
    if len(set(features)) != len(features):
        raise ValueError(f"`features` contains duplicated names: {features}.")
    return features


def _normalize_target(target: Any) -> str:
    if not isinstance(target, (str, bytes)):
        raise TypeError(
            f"`target` must be a single column name, received {type(target)!r}."
        )
    return str(target)


def _normalize_shards(shards: Any) -> List[str]:
    """Resolve the user-provided shards into an ordered list of files."""
    if isinstance(shards, (str, bytes, os.PathLike)):
        shards = [shards]

    resolved: List[str] = []
    for shard in shards:
        if isinstance(shard, os.PathLike):
            shard = os.fspath(shard)
        if not isinstance(shard, str):
            raise TypeError(
                "`shards` must contain file paths, received " f"{type(shard)!r}."
            )
        if any(char in shard for char in "*?["):
            matches = sorted(glob.glob(shard))
            if not matches:
                raise FileNotFoundError(
                    f"Shard pattern {shard!r} does not match any file."
                )
            resolved.extend(matches)
            continue
        if os.path.isdir(shard):
            matches = sorted(glob.glob(os.path.join(shard, "*.parquet")))
            if not matches:
                raise FileNotFoundError(
                    f"Directory {shard!r} does not contain Parquet shards."
                )
            resolved.extend(matches)
            continue
        resolved.append(shard)

    if len(resolved) == 0:
        raise ValueError("`shards` must contain at least one shard.")

    missing = [path for path in resolved if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Shard(s) not found: {missing}.")

    return resolved


def _sha256_file(path: os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _describe_callable(func: Optional[Callable]) -> dict:
    if func is None:
        return {"module": None, "qualname": None, "repr": None}
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    printable = False
    if module is not None and qualname is not None:
        try:
            loaded = _import_callable(module, qualname)
            printable = loaded is func
        except Exception:  # pragma: no cover - defensive
            printable = False
    return {
        "module": module,
        "qualname": qualname,
        "repr": repr(func)[:500],
        "importable": printable,
    }


def _import_callable(module: str, qualname: str) -> Callable:
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _load_callable(description: Optional[dict]) -> Optional[Callable]:
    if not description or not description.get("importable"):
        return None
    module = description.get("module")
    qualname = description.get("qualname")
    if not module or not qualname:  # pragma: no cover - defensive
        return None
    try:
        return _import_callable(module, qualname)
    except Exception:  # pragma: no cover - defensive
        return None


def _stored_scores_score(y_pred: Any, y_true: Any) -> None:
    """Placeholder score function for artifacts holding precomputed scores."""
    del y_pred, y_true
    raise CalibrationArtifactError(
        "This artifact stores precomputed nonconformity scores: its score "
        "function cannot be applied to new data."
    )


def _jsonable_or_none(value: Any) -> Any:
    """Serialize ``predict_kwargs`` for the artifact metadata."""
    if value is None:
        return None
    try:
        return json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "`predict_kwargs` must be JSON-serializable because they are "
            f"recorded in the artifact metadata. Received {value!r}."
        ) from exc


def _describe_predictor(predictor: Any) -> dict:
    model = getattr(predictor, "model", predictor)
    return {
        "predictor_class": type(predictor).__name__,
        "predictor_module": type(predictor).__module__,
        "predictor_repr": repr(predictor)[:500],
        "model_class": type(model).__name__,
        "model_module": type(model).__module__,
        "model_repr": repr(model)[:500],
    }


def _predict_batch(
    predictor: Any, X: np.ndarray, predict_kwargs: Optional[dict] = None
) -> np.ndarray:
    """Predict on one bounded batch and return a NumPy array."""
    predict_kwargs = dict(predict_kwargs or {})

    if isinstance(predictor, BasePredictor):
        raw_pred = predictor.predict(X, **predict_kwargs)
    elif hasattr(predictor, "predict"):
        raw_pred = predictor.predict(X, **predict_kwargs)
    elif callable(predictor):
        raw_pred = predictor(X)
    else:  # pragma: no cover - defensive
        raise TypeError(
            "`predictor` must be a BasePredictor, an object with a `predict` "
            f"method or a callable, received {type(predictor)!r}."
        )

    if raw_pred is None:
        raise CalibrationArtifactError(
            "The predictor returned None instead of predictions."
        )

    backend = get_backend(raw_pred)
    return np.asarray(backend.to_numpy(raw_pred))


def _scalar_scores(
    nonconf_score_func: Callable, y_pred: Any, y_true: np.ndarray, n_rows: int
) -> np.ndarray:
    """Compute one scalar nonconformity score per sample."""
    raw_scores = nonconf_score_func(y_pred, y_true)
    backend = get_backend(raw_scores)
    scores = np.asarray(backend.to_numpy(raw_scores))

    if not np.isrealobj(scores):
        raise CalibrationArtifactError(
            "Nonconformity scores must be real-valued, got complex values."
        )

    scores = np.squeeze(scores)
    if scores.ndim > 1:
        raise NotImplementedError(
            "Out-of-core calibration only supports scalar nonconformity "
            f"scores, received an array of shape {scores.shape}. For "
            "multivariate outputs, provide a scalar aggregation."
        )
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size != n_rows:
        raise CalibrationArtifactError(
            "Nonconformity score function must return one score per sample: "
            f"expected {n_rows}, received {scores.size}."
        )
    if np.any(np.isnan(scores)):
        raise CalibrationArtifactError(
            "Nonconformity scores contain NaN values. Out-of-core "
            "calibration requires finite or infinite scalar scores."
        )
    return scores


def _iter_parquet_batches(
    path: str, columns: Sequence[str], batch_size: int
) -> Iterable[Tuple[List[np.ndarray], List[str]]]:
    """Yield shard batches as ``(column arrays, arrow types)``.

    Arrays are returned in the order of ``columns``, i.e. the feature order
    requested by the caller, regardless of the column order of the shard.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading Parquet calibration shards requires `pyarrow`. Install "
            "it with `pip install pyarrow`."
        ) from exc

    try:
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:
        raise CalibrationArtifactError(
            f"Cannot read Parquet shard {path!r}: {exc}"
        ) from exc

    schema_names = list(parquet_file.schema_arrow.names)
    missing = [name for name in columns if name not in schema_names]
    if missing:
        raise CalibrationArtifactError(
            f"Shard {path!r} is missing column(s) {missing}. Available "
            f"columns: {schema_names}."
        )

    try:
        batches = parquet_file.iter_batches(
            batch_size=batch_size, columns=list(columns)
        )
        for batch in batches:
            index = {name: i for i, name in enumerate(batch.schema.names)}
            arrays = []
            types = []
            for name in columns:
                column = batch.column(index[name])
                if not (
                    pa.types.is_integer(column.type)
                    or pa.types.is_floating(column.type)
                    or pa.types.is_boolean(column.type)
                ):
                    raise CalibrationArtifactError(
                        f"Column {name!r} of shard {path!r} has non-numeric "
                        f"type {column.type}."
                    )
                arrays.append(column.to_numpy(zero_copy_only=False).reshape(-1))
                types.append(str(column.type))
            yield arrays, types
    except CalibrationArtifactError:
        raise
    except Exception as exc:
        raise CalibrationArtifactError(
            f"Error while reading shard {path!r}: {exc}"
        ) from exc


def _stack_features(feature_arrays: Sequence[np.ndarray]) -> np.ndarray:
    if len(feature_arrays) == 1:
        return np.asarray(feature_arrays[0]).reshape(-1, 1)
    return np.column_stack(feature_arrays)


class _RunReader:
    """Sequential buffered reader over one sorted run file."""

    def __init__(self, path: str, chunk_scores: int):
        self._path = path
        self._chunk_scores = max(1, int(chunk_scores))
        self._file = open(path, "rb")
        self._buffer = np.empty(0, dtype=SCORES_DTYPE)
        self._position = 0

    def _refill(self) -> bool:
        raw = self._file.read(self._chunk_scores * BYTES_PER_SCORE)
        if not raw:
            self._buffer = np.empty(0, dtype=SCORES_DTYPE)
            self._position = 0
            return False
        self._buffer = np.frombuffer(raw, dtype=SCORES_DTYPE)
        self._position = 0
        return True

    def ensure_loaded(self) -> bool:
        """Make sure the next score is buffered; return ``False`` if done."""
        while self._position >= self._buffer.size:
            if not self._refill():
                return False
        return True

    def peek(self) -> float:
        """Current (smallest unread) score of the run."""
        return float(self._buffer[self._position])

    def take(self, max_count: int, limit: Optional[float]) -> Optional[np.ndarray]:
        """Take at most ``max_count`` scores whose values are ``<= limit``.

        At least one score is returned when the run is not exhausted, even if
        it exceeds ``limit`` (the caller re-pushes the run on its heap).
        """
        parts = []
        taken = 0
        while taken < max_count:
            if not self.ensure_loaded():
                break
            available = self._buffer[self._position :]
            if limit is None:
                count = int(min(available.size, max_count - taken))
            else:
                count = int(np.searchsorted(available, limit, side="right"))
                if count == 0:
                    if taken == 0:
                        count = 1
                    else:
                        break
                count = int(min(count, max_count - taken))
            parts.append(available[:count])
            self._position += count
            taken += count
            if taken >= max_count:
                break
            if self._position < self._buffer.size and limit is not None:
                if self._buffer[self._position] > limit:
                    break
        if taken == 0:
            return None
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts)

    def close(self) -> None:
        self._file.close()


def _external_merge(
    run_paths: Sequence[str],
    output_path: str,
    chunk_scores: int,
    buffer_scores: int,
) -> None:
    """K-way merge sorted runs into a single sorted file, in bounded memory."""
    readers = [_RunReader(path, chunk_scores) for path in run_paths]
    heap: List[Tuple[float, int]] = []
    try:
        for index, reader in enumerate(readers):
            if reader.ensure_loaded():
                heapq.heappush(heap, (reader.peek(), index))

        max_block = max(1, int(buffer_scores))
        with open(output_path, "wb") as output:
            while heap:
                _, index = heapq.heappop(heap)
                reader = readers[index]
                second = heap[0][0] if heap else None
                block = reader.take(max_block, second)
                if block is None:  # pragma: no cover - defensive
                    continue
                output.write(np.ascontiguousarray(block, dtype=SCORES_DTYPE).tobytes())
                if reader.ensure_loaded():
                    heapq.heappush(heap, (reader.peek(), index))
    finally:
        for reader in readers:
            reader.close()


def _reduce_runs(
    run_paths: Sequence[str],
    run_dir: str,
    max_fan_in: int,
    chunk_scores: int,
    buffer_scores: int,
) -> str:
    """Merge runs in several passes so that open files stay bounded."""
    max_fan_in = max(2, int(max_fan_in))
    current = list(run_paths)
    pass_index = 0
    while len(current) > 1:
        merged: List[str] = []
        for group_start in range(0, len(current), max_fan_in):
            group = current[group_start : group_start + max_fan_in]
            if len(group) == 1:
                merged.append(group[0])
                continue
            output = os.path.join(
                run_dir, f"merge-{pass_index:03d}-{len(merged):06d}.f8"
            )
            _external_merge(group, output, chunk_scores, buffer_scores)
            for path in group:
                os.remove(path)
            merged.append(output)
        current = merged
        pass_index += 1
    return current[0]


def _write_run(buffer: np.ndarray, run_dir: str, run_index: int) -> str:
    path = os.path.join(run_dir, f"run-{run_index:06d}.f8")
    with open(path, "wb") as file_handle:
        buffer.sort()
        file_handle.write(np.ascontiguousarray(buffer, dtype=SCORES_DTYPE).tobytes())
    return path


def _order_statistic_index(total_scores: int, quantile_level: float) -> int:
    """Reproduce NumPy's ``inverted_cdf`` index on ``total_scores`` values."""
    virtual_index = float(total_scores) * quantile_level - 1.0
    previous = np.floor(virtual_index)
    gamma = virtual_index - previous
    index = previous if gamma == 0 else previous + 1
    index = int(max(index, 0))
    return min(index, total_scores - 1)


def _normalize_alpha(alpha: Any) -> float:
    """Validate that ``alpha`` is a real scalar and return it as a float."""
    if isinstance(alpha, bool):
        raise NotImplementedError(
            "Out-of-core calibration only supports scalar significance "
            f"levels, received {type(alpha)!r}. Multivariate calibration is "
            "not supported."
        )
    if isinstance(alpha, np.ndarray):
        if alpha.ndim != 0:
            raise NotImplementedError(
                "Out-of-core calibration only supports scalar significance "
                f"levels, received an array of shape {alpha.shape}. "
                "Multivariate calibration is not supported."
            )
        alpha = alpha.item()
    if not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise NotImplementedError(
            "Out-of-core calibration only supports scalar significance "
            f"levels, received {type(alpha)!r}. Multivariate calibration is "
            "not supported."
        )
    return float(alpha)


class _SortedScoreFile:
    """Random access to a sorted score file without loading it in memory.

    The underlying file handle is opened lazily and is not serialized, so the
    owning artifact can be deep-copied or pickled (as done by
    :class:`~deel.puncc.api.conformalization.CrossValCpAggregator` and
    :meth:`ConformalPredictor.save`).
    """

    def __init__(self, path: str, count: int):
        self._path = str(path)
        self._count = int(count)
        self._file: Optional[Any] = None

    def __getstate__(self) -> dict:
        return {"path": self._path, "count": self._count}

    def __setstate__(self, state: dict) -> None:
        self._path = state["path"]
        self._count = int(state["count"])
        self._file = None

    def __len__(self) -> int:
        return self._count

    def _get_file(self):
        if self._file is None or self._file.closed:
            try:
                self._file = open(self._path, "rb")
            except OSError as exc:
                raise ArtifactNotReadyError(
                    f"Cannot open the artifact score file {self._path!r}: " f"{exc}"
                ) from exc
        return self._file

    def close(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()
        self._file = None

    def score_at(self, index: int) -> float:
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(
                f"Score index {index} is out of range for {self._count} " "scores."
            )
        file_handle = self._get_file()
        file_handle.seek(index * BYTES_PER_SCORE)
        raw = file_handle.read(BYTES_PER_SCORE)
        if len(raw) != BYTES_PER_SCORE:
            raise ArtifactNotReadyError(
                "Score file is truncated: cannot read the requested score."
            )
        return float(np.frombuffer(raw, dtype=SCORES_DTYPE)[0])

    def read(self, start: int = 0, stop: Optional[int] = None) -> np.ndarray:
        start = int(start)
        stop = self._count if stop is None else int(stop)
        if start < 0:
            start += self._count
        if stop < 0:
            stop += self._count
        if start < 0 or stop > self._count or stop < start:
            raise IndexError(
                f"Invalid score range [{start}, {stop}) for {self._count} " "scores."
            )
        count = stop - start
        file_handle = self._get_file()
        file_handle.seek(start * BYTES_PER_SCORE)
        raw = file_handle.read(count * BYTES_PER_SCORE)
        if len(raw) != count * BYTES_PER_SCORE:
            raise ArtifactNotReadyError(
                "Score file is truncated: cannot read the requested range."
            )
        return np.frombuffer(raw, dtype=SCORES_DTYPE).copy()

    def iter_chunks(self, chunk_scores: int = DEFAULT_MERGE_CHUNK_SCORES):
        chunk_scores = max(1, int(chunk_scores))
        start = 0
        while start < self._count:
            stop = min(start + chunk_scores, self._count)
            yield self.read(start, stop)
            start = stop


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Artifact ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class OutOfCoreCalibrationArtifact:
    """A reopened, disk-backed conformal calibration artifact.

    Instances are created by :meth:`build` or :meth:`open` and should be
    closed after use. They expose the same calibration interface as
    :class:`~deel.puncc.api.calibration.BaseCalibrator` through
    :meth:`compute_quantile` and :meth:`calibrate`.

    :param path: artifact directory.
    :param metadata: decoded metadata dictionary.
    :param verify_checksum: if ``True``, the sorted score file is hashed on
        opening.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        metadata: dict,
        verify_checksum: bool = False,
    ):
        self.path = Path(path)
        self.metadata = metadata
        self._closed = False

        scores_info = metadata.get("scores", {})
        count = int(metadata["n_samples"])
        scores_path = self.path / scores_info.get("file", SCORES_FILE)
        if not scores_path.is_file():
            raise ArtifactNotReadyError(f"Score file {str(scores_path)!r} is missing.")
        if verify_checksum:
            expected = scores_info.get("sha256")
            if expected is not None and _sha256_file(scores_path) != expected:
                raise ArtifactNotReadyError(
                    f"Score file {scores_path} checksum mismatch."
                )
        self._scores = _SortedScoreFile(str(scores_path), count)

        score_description = metadata.get("nonconformity_score")
        self.nonconf_score_func = _load_callable(score_description)
        if self.nonconf_score_func is None:
            self.nonconf_score_func = _stored_scores_score

        self.features: Tuple[str, ...] = tuple(metadata.get("features", ()))
        self.target: Optional[str] = metadata.get("target")
        self.batch_size: Optional[int] = metadata.get("batch_size")
        stored_kwargs = metadata.get("predict_kwargs")
        self.predict_kwargs: Optional[dict] = (
            dict(stored_kwargs) if isinstance(stored_kwargs, dict) else None
        )
        self._predictor = None

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return (
            f"{self.__class__.__name__}(path={str(self.path)!r}, "
            f"n_samples={len(self)}, status={status})"
        )

    def __enter__(self) -> "OutOfCoreCalibrationArtifact":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._scores)

    @property
    def n_samples(self) -> int:
        """Number of stored calibration scores (duplicates included)."""
        return len(self)

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise CalibrationArtifactError(
                f"Artifact {str(self.path)!r} is closed. Reopen it with "
                "`open_calibration_artifact`."
            )

    @classmethod
    def build(
        cls,
        shards: Iterable[Union[str, os.PathLike]],
        features: Optional[Iterable[str]] = None,
        target: str = "y",
        predictor: Any = None,
        *,
        path: Optional[Union[str, os.PathLike]] = None,
        artifact_dir: Optional[Union[str, os.PathLike]] = None,
        nonconf_score_func: Optional[Callable] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sort_buffer_scores: int = DEFAULT_SORT_BUFFER_SCORES,
        merge_chunk_scores: int = DEFAULT_MERGE_CHUNK_SCORES,
        max_fan_in: int = DEFAULT_MAX_FAN_IN,
        predict_kwargs: Optional[dict] = None,
        save_predictor: bool = False,
        overwrite: bool = False,
        verify_checksum: bool = False,
    ) -> "OutOfCoreCalibrationArtifact":
        """Build an out-of-core calibration artifact from Parquet shards.

        All shards are streamed batch by batch, in the provided order, and the
        given feature columns are read in the provided order. Predictions are
        computed with the fixed, already trained ``predictor``; no training is
        performed. Scalar nonconformity scores are written to a temporary run
        buffer, sorted and spilled to disk when the buffer is full, then
        k-way merged into a single sorted score file.

        The artifact is assembled in a temporary directory and atomically moved
        to ``path`` once the score file, metadata and completion marker have
        been written and flushed. A failed build never exposes a usable
        artifact.

        :param shards: ordered list of Parquet shards. A single path, a
            directory containing ``*.parquet`` files, or a glob pattern is also
            accepted.
        :param features: ordered feature column names. If ``None``, the
            columns of the first shard (except the target) are used, preserving
            their file order.
        :param target: target column name.
        :param predictor: already trained predictor exposing ``predict``.
        :param path: destination directory of the artifact.
        :param artifact_dir: alias of ``path``.
        :param nonconf_score_func: scalar nonconformity score function taking
            ``(y_pred, y_true)``. Defaults to
            :func:`deel.puncc.api.nonconformity_scores.absolute_difference`.
        :param batch_size: maximum number of samples pushed to the predictor
            at once. It also bounds the size of the Parquet batches.
        :param sort_buffer_scores: maximum number of scores kept in memory
            before spilling a sorted run to disk.
        :param merge_chunk_scores: number of scores read at once per run during
            the final k-way merge. This bounds the merge memory.
        :param max_fan_in: maximum number of runs merged at once. Larger run
            sets are merged in several passes so that the number of open files
            and the heap stay bounded.
        :param predict_kwargs: optional keyword arguments forwarded to the
            predictor's ``predict`` method.
        :param save_predictor: if ``True``, pickle the predictor inside the
            artifact to enable :meth:`predict` later on.
        :param overwrite: if ``True``, replace an existing artifact directory.
        :param verify_checksum: if ``True``, verify the score file checksum
            before returning the opened artifact.

        :returns: the opened artifact.
        :rtype: OutOfCoreCalibrationArtifact

        :raises CalibrationArtifactError: build failure.
        :raises FileNotFoundError: missing shard.
        :raises ValueError: invalid arguments.
        """
        if path is None and artifact_dir is None:
            raise ValueError("Provide `path` (or its alias `artifact_dir`).")
        if path is not None and artifact_dir is not None:
            raise ValueError("Provide either `path` or `artifact_dir`, not both.")
        destination = Path(path if path is not None else artifact_dir)

        if predictor is None:
            raise ValueError("A trained `predictor` is required.")
        if nonconf_score_func is None:
            nonconf_score_func = nonconformity_scores.absolute_difference
        if batch_size < 1:
            raise ValueError("`batch_size` must be a positive integer.")
        if sort_buffer_scores < 1:
            raise ValueError("`sort_buffer_scores` must be a positive integer.")
        if merge_chunk_scores < 1:
            raise ValueError("`merge_chunk_scores` must be a positive integer.")
        if max_fan_in < 2:
            raise ValueError("`max_fan_in` must be an integer >= 2.")

        shard_paths = _normalize_shards(shards)
        target = _normalize_target(target)

        if features is None:
            features = _infer_features(shard_paths[0], target)
        feature_names = _normalize_features(features)
        columns = feature_names + [target]

        if destination.exists():
            if not overwrite:
                raise CalibrationArtifactError(
                    f"Artifact destination {str(destination)!r} already "
                    "exists. Pass `overwrite=True` to replace it."
                )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.building-",
                dir=str(destination.parent),
            )
        )

        try:
            metadata = {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "status": "building",
                "created_at": _utc_now(),
                "features": feature_names,
                "target": target,
                "batch_size": int(batch_size),
                "sort_buffer_scores": int(sort_buffer_scores),
                "merge_chunk_scores": int(merge_chunk_scores),
                "max_fan_in": int(max_fan_in),
                "predict_kwargs": _jsonable_or_none(predict_kwargs),
                "nonconformity_score": _describe_callable(nonconf_score_func),
                "predictor": _describe_predictor(predictor),
            }
            _write_json(temporary / METADATA_FILE, metadata)

            n_samples, feature_dtypes, target_dtype, rows_per_shard = _stream_scores(
                shard_paths=shard_paths,
                columns=columns,
                target=target,
                predictor=predictor,
                nonconf_score_func=nonconf_score_func,
                batch_size=batch_size,
                sort_buffer_scores=sort_buffer_scores,
                merge_chunk_scores=merge_chunk_scores,
                max_fan_in=max_fan_in,
                predict_kwargs=predict_kwargs,
                workdir=temporary,
            )

            scores_path = temporary / SCORES_FILE
            _fsync_file(scores_path)
            scores_sha256 = _sha256_file(scores_path)

            predictor_file = None
            if save_predictor:
                predictor_file = MODEL_FILE
                try:
                    with open(temporary / MODEL_FILE, "wb") as handle:
                        pickle.dump(predictor, handle, protocol=4)
                except Exception as exc:
                    raise CalibrationArtifactError(
                        "Cannot pickle the predictor. Rebuild with "
                        "`save_predictor=False` to store only a descriptor of "
                        f"the model. Original error: {exc}"
                    ) from exc

            metadata.update(
                {
                    "status": "complete",
                    "completed_at": _utc_now(),
                    "n_samples": int(n_samples),
                    "scores": {
                        "file": SCORES_FILE,
                        "dtype": "float64",
                        "byteorder": "little",
                        "count": int(n_samples),
                        "sha256": scores_sha256,
                    },
                    "input_schema": {
                        "features": feature_names,
                        "feature_dtypes": feature_dtypes,
                        "target": target,
                        "target_dtype": target_dtype,
                    },
                    "shards": [
                        {
                            "path": shard_path,
                            "rows": rows,
                            "size_bytes": os.path.getsize(shard_path),
                        }
                        for shard_path, rows in zip(shard_paths, rows_per_shard)
                    ],
                    "model": {
                        **_describe_predictor(predictor),
                        "pickled_file": predictor_file,
                    },
                    "infinite_score_convention": (
                        "one +inf score is appended to the empirical "
                        "distribution, following Tibshirani's lemma"
                    ),
                }
            )
            metadata_path = temporary / METADATA_FILE
            _write_json(metadata_path, metadata)
            _fsync_file(metadata_path)

            marker = {
                "format": FORMAT_NAME,
                "format_version": FORMAT_VERSION,
                "n_samples": int(n_samples),
                "scores_sha256": scores_sha256,
                "metadata_sha256": _sha256_file(metadata_path),
            }
            _write_marker(temporary / COMPLETE_MARKER, marker)
            _fsync_directory(temporary)

            backup = None
            if destination.exists():
                backup = Path(
                    tempfile.mkdtemp(
                        prefix=f".{destination.name}.previous-",
                        dir=str(destination.parent),
                    )
                )
                backup = backup / "artifact"
                os.replace(str(destination), str(backup))
                _fsync_directory(destination.parent)
            try:
                os.replace(str(temporary), str(destination))
            except BaseException:
                if backup is not None and backup.exists():
                    os.replace(str(backup), str(destination))
                raise
            if backup is not None and backup.exists():
                if backup.is_dir() and not backup.is_symlink():
                    shutil.rmtree(backup, ignore_errors=True)
                else:  # pragma: no cover - defensive
                    backup.unlink()
                shutil.rmtree(backup.parent, ignore_errors=True)
            _fsync_directory(destination.parent)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        return cls(destination, metadata, verify_checksum=verify_checksum)

    @classmethod
    def open(
        cls,
        path: Union[str, os.PathLike],
        *,
        verify_checksum: bool = False,
    ) -> "OutOfCoreCalibrationArtifact":
        """Reopen an existing artifact without rescanning the source shards.

        The artifact must be complete: the completion marker, the metadata and
        a score file whose size matches the recorded number of samples are
        required. Anything else raises :class:`ArtifactNotReadyError`, so an
        interrupted build can never be mistaken for a usable calibrator.

        :param path: artifact directory.
        :param verify_checksum: if ``True``, hash the score file.

        :returns: the opened artifact.
        :rtype: OutOfCoreCalibrationArtifact

        :raises ArtifactNotReadyError: missing, incomplete or corrupted
            artifact.
        """
        directory = Path(path)
        if not directory.is_dir():
            raise ArtifactNotReadyError(
                f"Artifact path {str(directory)!r} is not a directory."
            )

        metadata_path = directory / METADATA_FILE
        if not metadata_path.is_file():
            raise ArtifactNotReadyError(
                f"Artifact {str(directory)!r} has no {METADATA_FILE}: it is "
                "incomplete or is not a puncc calibration artifact."
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactNotReadyError(
                f"Artifact metadata {metadata_path!r} cannot be read: {exc}"
            ) from exc

        _validate_metadata(metadata, directory)

        marker_path = directory / COMPLETE_MARKER
        if not marker_path.is_file():
            raise ArtifactNotReadyError(
                f"Artifact {str(directory)!r} is incomplete: the completion "
                "marker is missing. Rebuild the artifact."
            )
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ArtifactNotReadyError(
                f"Artifact completion marker {str(marker_path)!r} cannot be "
                f"read: {exc}"
            ) from exc

        if marker.get("format") != FORMAT_NAME:
            raise ArtifactFormatError(
                f"Completion marker format {marker.get('format')!r} is not "
                f"{FORMAT_NAME!r}."
            )
        if marker.get("format_version") != FORMAT_VERSION:
            raise ArtifactFormatError(
                "Unsupported artifact format version "
                f"{marker.get('format_version')!r}; expected "
                f"{FORMAT_VERSION}."
            )
        if (
            metadata["scores"].get("sha256") is not None
            and marker.get("scores_sha256") != metadata["scores"]["sha256"]
        ):
            raise ArtifactNotReadyError(
                "Artifact is inconsistent: the completion marker does not "
                "match the metadata score checksum."
            )
        if marker.get("metadata_sha256") != _sha256_file(metadata_path):
            raise ArtifactNotReadyError(
                "Artifact is inconsistent: metadata was modified after the "
                "completion marker was written."
            )

        scores_info = metadata["scores"]
        scores_path = directory / scores_info.get("file", SCORES_FILE)
        if not scores_path.is_file():
            raise ArtifactNotReadyError(
                f"Artifact {str(directory)!r} has no score file "
                f"{scores_info.get('file', SCORES_FILE)!r}."
            )
        expected_size = int(metadata["n_samples"]) * BYTES_PER_SCORE
        actual_size = scores_path.stat().st_size
        if actual_size != expected_size:
            raise ArtifactNotReadyError(
                f"Score file {str(scores_path)!r} is corrupted: expected "
                f"{expected_size} bytes for {metadata['n_samples']} scores, "
                f"found {actual_size}."
            )

        return cls(directory, metadata, verify_checksum=verify_checksum)

    load = open

    def close(self) -> None:
        """Close the artifact. It can be reopened with :meth:`open`."""
        if not self._closed:
            self._scores.close()
            self._closed = True

    # ----- score access -----

    def score_at(self, index: int) -> float:
        """Return the ``index``-th smallest stored score."""
        self._ensure_open()
        return self._scores.score_at(index)

    def get_scores(self, start: int = 0, stop: Optional[int] = None) -> np.ndarray:
        """Load a bounded slice of the sorted scores.

        This method is intended for inspection; the calibration path below
        never loads all scores at once.
        """
        self._ensure_open()
        return self._scores.read(start, stop)

    def iter_score_chunks(
        self, chunk_scores: int = DEFAULT_MERGE_CHUNK_SCORES
    ) -> Iterable[np.ndarray]:
        """Iterate over the sorted scores in bounded chunks."""
        self._ensure_open()
        return self._scores.iter_chunks(chunk_scores)

    # ----- calibration interface -----

    def compute_quantile(
        self,
        *,
        alpha: float,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = bonferroni,
    ) -> np.ndarray:
        """Compute the ``1 - alpha`` empirical quantile of the stored scores.

        The convention of :class:`~deel.puncc.api.calibration.BaseCalibrator`
        is preserved: one ``+inf`` score is appended to the ``n`` stored
        scores and ``alpha`` must satisfy ``1 / (n + 1) <= alpha < 1``.

        :param float alpha: significance level.
        :param weights: unsupported for out-of-core calibration.
        :param Callable correction: correction for multiple hypothesis
            testing. Defaults to Bonferroni.

        :returns: exact finite-sample quantile.
        :rtype: np.ndarray

        :raises ValueError: invalid ``alpha``.
        :raises NotImplementedError: weighted or multivariate calibration.
        """
        self._ensure_open()

        if weights is not None:
            raise NotImplementedError(
                "Weighted conformal calibration is not supported by "
                "out-of-core artifacts: calibration weights cannot be stored "
                "without keeping one weight per sample. Use a regular "
                "BaseCalibrator instead."
            )

        alpha = _normalize_alpha(alpha)

        if correction is not None:
            corrected = correction(alpha)
            if not np.isscalar(corrected) and np.ndim(corrected) != 0:
                raise NotImplementedError(
                    "The correction function returned a non-scalar "
                    "significance level; multivariate calibration is not "
                    "supported by out-of-core artifacts."
                )
            alpha = float(corrected)

        alpha_calib_check(alpha=alpha, n=len(self))

        total_scores = len(self) + 1  # appended +inf score
        quantile_level = 1.0 - alpha
        index = _order_statistic_index(total_scores, quantile_level)

        if index >= len(self):
            return np.asarray(float("inf"))

        return np.asarray(self._scores.score_at(index))

    quantile = compute_quantile

    def calibrate(
        self,
        *,
        alpha: float,
        X: Optional[Iterable] = None,
        y_pred: Iterable,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = bonferroni,
        pred_set_func: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, ...]:
        """Build prediction sets for new predictions, as ``BaseCalibrator``.

        :param float alpha: significance level.
        :param X: optional features (unused unless the prediction set function
            needs them).
        :param y_pred: predictions for the new samples.
        :param weights: unsupported for out-of-core calibration.
        :param Callable correction: correction for multiple hypothesis
            testing. Defaults to Bonferroni.
        :param Callable pred_set_func: prediction set function; defaults to the
            artifact's prediction set function (constant interval).

        :returns: prediction sets (e.g. ``(y_lower, y_upper)``).
        :rtype: Tuple[np.ndarray, ...]
        """
        del X
        score_quantile = self.compute_quantile(
            alpha=alpha, weights=weights, correction=correction
        )
        set_function = pred_set_func if pred_set_func is not None else constant_interval
        return set_function(y_pred, scores_quantile=score_quantile)

    def to_calibrator(
        self, *, pred_set_func: Optional[Callable] = None
    ) -> "OutOfCoreCalibrator":
        """Return a :class:`BaseCalibrator`-compatible calibrator.

        The returned calibrator uses the artifact as its score store: it
        implements ``compute_quantile`` and ``calibrate`` exactly like the
        in-memory calibrators, and can therefore be used anywhere a calibrated
        quantile or a prediction set is needed.

        :param Callable pred_set_func: optional prediction set function.
        :returns: artifact-backed calibrator.
        :rtype: OutOfCoreCalibrator
        """
        self._ensure_open()
        return OutOfCoreCalibrator(artifact=self, pred_set_func=pred_set_func)

    make_calibrator = to_calibrator

    @property
    def predictor_descriptor(self) -> dict:
        """Descriptor of the predictor bound at build time."""
        return dict(self.metadata.get("model", {}))

    @property
    def predictor(self):
        """The predictor pickled at build time, if any.

        :raises CalibrationArtifactError: the predictor was not saved.
        """
        self._ensure_open()
        if self._predictor is not None:
            return self._predictor
        model_info = self.metadata.get("model", {})
        pickle_name = model_info.get("pickled_file")
        if not pickle_name:
            raise CalibrationArtifactError(
                "This artifact does not embed the predictor. Rebuild it with "
                "`save_predictor=True` to use `predict`."
            )
        model_path = self.path / pickle_name
        if not model_path.is_file():  # pragma: no cover - defensive
            raise ArtifactNotReadyError(
                f"Predictor file {str(model_path)!r} is missing."
            )
        with open(model_path, "rb") as file_handle:
            self._predictor = pickle.load(file_handle)
        return self._predictor

    def predict(
        self,
        X: Iterable,
        alpha: float,
        *,
        correction: Optional[Callable] = bonferroni,
        pred_set_func: Optional[Callable] = None,
        predict_kwargs: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict and calibrate intervals with the bound predictor.

        This is the out-of-core equivalent of ``SplitCP.predict``: the
        prediction is computed by the predictor saved in the artifact and the
        interval uses the exact quantile of the stored scores. The source
        shards are never read again.

        :param X: new features.
        :param float alpha: significance level.
        :param Callable correction: correction for multiple hypothesis
            testing.
        :param Callable pred_set_func: prediction set function.
        :param dict predict_kwargs: keyword arguments forwarded to the
            predictor. Defaults to the keyword arguments recorded at build
            time.

        :returns: ``(y_pred, y_lower, y_upper)``.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        self._ensure_open()
        predictor = self.predictor
        if predict_kwargs is None:
            predict_kwargs = self.predict_kwargs
        y_pred = _predict_batch(predictor, X, predict_kwargs)
        y_pred = np.squeeze(y_pred)
        score_quantile = self.compute_quantile(alpha=alpha, correction=correction)
        set_function = pred_set_func if pred_set_func is not None else constant_interval
        set_pred = set_function(y_pred, scores_quantile=score_quantile)
        return (y_pred, *set_pred)

    predict_intervals = predict


class OutOfCoreCalibrator(BaseCalibrator):
    """Artifact-backed :class:`BaseCalibrator`.

    It behaves like an in-memory calibrator for the purpose of building
    prediction sets, but its score distribution lives on disk. Calling
    :meth:`fit` is not supported: the scores were computed once, at artifact
    build time.

    :param OutOfCoreCalibrationArtifact artifact: opened artifact.
    :param Callable pred_set_func: prediction set function. Defaults to
        :func:`deel.puncc.api.prediction_sets.constant_interval`.
    """

    def __init__(
        self,
        artifact: OutOfCoreCalibrationArtifact,
        *,
        pred_set_func: Optional[Callable] = None,
    ):
        super().__init__(
            nonconf_score_func=artifact.nonconf_score_func,
            pred_set_func=pred_set_func or constant_interval,
            weight_func=None,
        )
        self.artifact = artifact
        self._len_calib = len(artifact)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(path={str(self.artifact.path)!r}, "
            f"n_samples={self._len_calib})"
        )

    def fit(self, *args, **kwargs) -> None:  # pylint: disable=arguments-differ
        """Not supported: scores are computed when the artifact is built."""
        del args, kwargs
        raise RuntimeError(
            "OutOfCoreCalibrator cannot be fitted: its nonconformity scores "
            "are stored in the artifact. Build the artifact with "
            "`OutOfCoreCalibrationArtifact.build` instead."
        )

    def get_nonconformity_scores(self) -> None:
        """Return ``None``: scores are not loaded in memory."""
        return None

    def compute_quantile(
        self,
        *,
        alpha: float,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = bonferroni,
    ) -> np.ndarray:
        """Compute the calibrated quantile from the artifact."""
        return self.artifact.compute_quantile(
            alpha=alpha, weights=weights, correction=correction
        )

    def calibrate(
        self,
        *,
        alpha: float,
        X: Optional[Iterable] = None,
        y_pred: Iterable,
        weights: Optional[Iterable] = None,
        correction: Optional[Callable] = bonferroni,
    ) -> Tuple[np.ndarray, ...]:
        """Compute calibrated prediction sets from the artifact."""
        del X
        return super().calibrate(
            alpha=alpha,
            y_pred=y_pred,
            weights=weights,
            correction=correction,
        )


# Aliases kept for discoverability.
CalibrationArtifact = OutOfCoreCalibrationArtifact
ParquetCalibrationArtifact = OutOfCoreCalibrationArtifact


def build_calibration_artifact(*args, **kwargs) -> OutOfCoreCalibrationArtifact:
    """Functional alias of :meth:`OutOfCoreCalibrationArtifact.build`."""
    return OutOfCoreCalibrationArtifact.build(*args, **kwargs)


def open_calibration_artifact(
    path: Union[str, os.PathLike], **kwargs
) -> OutOfCoreCalibrationArtifact:
    """Functional alias of :meth:`OutOfCoreCalibrationArtifact.open`."""
    return OutOfCoreCalibrationArtifact.open(path, **kwargs)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Build internals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _infer_features(shard_path: str, target: str) -> List[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Reading Parquet calibration shards requires `pyarrow`. Install "
            "it with `pip install pyarrow`."
        ) from exc
    names = list(pq.ParquetFile(shard_path).schema_arrow.names)
    if target not in names:
        raise CalibrationArtifactError(
            f"Shard {shard_path!r} is missing the target column {target!r}. "
            f"Available columns: {names}."
        )
    features = [name for name in names if name != target]
    if not features:
        raise CalibrationArtifactError(
            f"Shard {shard_path!r} has no feature column besides the target."
        )
    return features


def _stream_scores(
    *,
    shard_paths: Sequence[str],
    columns: Sequence[str],
    target: str,
    predictor: Any,
    nonconf_score_func: Callable,
    batch_size: int,
    sort_buffer_scores: int,
    merge_chunk_scores: int,
    max_fan_in: int,
    predict_kwargs: Optional[dict],
    workdir: Path,
) -> Tuple[int, List[str], Optional[str], List[int]]:
    """Stream all shards and write the sorted score file in ``workdir``.

    Returns ``(n_samples, feature_dtypes, target_dtype, rows_per_shard)``.
    """
    run_dir = workdir / "runs"
    run_dir.mkdir()

    buffer = np.empty(sort_buffer_scores, dtype=np.float64)
    buffer_fill = 0
    run_paths: List[str] = []

    n_samples = 0
    rows_per_shard: List[int] = []
    feature_dtypes: Optional[List[str]] = None
    target_dtype: Optional[str] = None

    def flush_run() -> None:
        nonlocal buffer_fill
        if buffer_fill == 0:
            return
        path = _write_run(buffer[:buffer_fill], str(run_dir), len(run_paths))
        run_paths.append(path)
        buffer_fill = 0

    for shard_path in shard_paths:
        shard_rows = 0
        for arrays, types in _iter_parquet_batches(shard_path, columns, batch_size):
            if len(arrays) != len(columns):  # pragma: no cover - defensive
                raise CalibrationArtifactError(
                    f"Unexpected batch layout while reading {shard_path!r}."
                )
            feature_arrays = arrays[:-1]
            target_array = arrays[-1]
            if feature_arrays[0].size == 0:
                continue

            feature_types = types[:-1]
            current_target_type = types[-1]
            if feature_dtypes is None:
                feature_dtypes = feature_types
                target_dtype = current_target_type
            elif feature_types != feature_dtypes or current_target_type != target_dtype:
                raise CalibrationArtifactError(
                    "Inconsistent column types across shards: expected "
                    f"features {feature_dtypes} and target {target_dtype}, "
                    f"found features {feature_types} and target "
                    f"{current_target_type} in shard {shard_path!r}."
                )

            X = _stack_features(feature_arrays)
            y_true = np.asarray(target_array).reshape(-1)
            n_rows = X.shape[0]
            if y_true.size != n_rows:
                raise CalibrationArtifactError(
                    f"Target column {target!r} of shard {shard_path!r} does "
                    f"not match the feature rows ({y_true.size} vs {n_rows})."
                )

            y_pred = _predict_batch(predictor, X, predict_kwargs)
            scores = _scalar_scores(nonconf_score_func, y_pred, y_true, n_rows)

            offset = 0
            while offset < scores.size:
                take = min(scores.size - offset, buffer.size - buffer_fill)
                buffer[buffer_fill : buffer_fill + take] = scores[
                    offset : offset + take
                ]
                buffer_fill += take
                offset += take
                if buffer_fill == buffer.size:
                    flush_run()

            n_samples += n_rows
            shard_rows += n_rows
        rows_per_shard.append(shard_rows)

    flush_run()

    if n_samples == 0:
        raise CalibrationArtifactError(
            "No calibration sample was read from the provided shards."
        )
    if feature_dtypes is None or target_dtype is None:  # pragma: no cover
        raise CalibrationArtifactError("Could not infer the input schema.")

    scores_path = workdir / SCORES_FILE
    if len(run_paths) == 1:
        os.replace(run_paths[0], scores_path)
    else:
        merged_path = _reduce_runs(
            run_paths,
            str(run_dir),
            max_fan_in=max_fan_in,
            chunk_scores=merge_chunk_scores,
            buffer_scores=merge_chunk_scores,
        )
        os.replace(merged_path, scores_path)

    shutil.rmtree(run_dir, ignore_errors=True)

    if scores_path.stat().st_size != n_samples * BYTES_PER_SCORE:
        raise CalibrationArtifactError(
            "Internal error: sorted score file size does not match the "
            f"number of calibration samples ({n_samples})."
        )

    return n_samples, feature_dtypes, target_dtype, rows_per_shard


def _write_json(path: Path, payload: dict) -> None:
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    with open(path, "wb") as file_handle:
        file_handle.write(raw)
        file_handle.flush()
        os.fsync(file_handle.fileno())


def _write_marker(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        _write_json(temporary, payload)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():  # pragma: no cover - defensive
            temporary.unlink()


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as file_handle:
        os.fsync(file_handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        file_descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _validate_metadata(metadata: dict, directory: Path) -> None:
    if metadata.get("format") != FORMAT_NAME:
        raise ArtifactFormatError(
            f"Artifact {str(directory)!r} has format "
            f"{metadata.get('format')!r}, expected {FORMAT_NAME!r}."
        )
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ArtifactFormatError(
            "Unsupported artifact format version "
            f"{metadata.get('format_version')!r}; expected {FORMAT_VERSION}."
        )
    if metadata.get("status") != "complete":
        raise ArtifactNotReadyError(
            f"Artifact {str(directory)!r} is not complete (status "
            f"{metadata.get('status')!r}). Rebuild it."
        )
    if "n_samples" not in metadata or int(metadata["n_samples"]) < 1:
        raise ArtifactNotReadyError(
            f"Artifact {str(directory)!r} records no calibration sample."
        )
    if not isinstance(metadata.get("scores"), dict):
        raise ArtifactFormatError(
            f"Artifact {str(directory)!r} metadata has no score description."
        )
