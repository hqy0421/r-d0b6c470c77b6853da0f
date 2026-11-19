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
import copy
import json
import os
import pickle

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from deel.puncc.api import nonconformity_scores  # noqa: E402
from deel.puncc.api.calibration import BaseCalibrator  # noqa: E402
from deel.puncc.api.out_of_core import (  # noqa: E402
    COMPLETE_MARKER,
    METADATA_FILE,
    SCORES_FILE,
    ArtifactNotReadyError,
    CalibrationArtifactError,
    OutOfCoreCalibrator,
    build_calibration_artifact,
    open_calibration_artifact,
)
from deel.puncc.api.prediction_sets import constant_interval  # noqa: E402

FEATURES = ["f0", "f1"]


class RecordingPredictor:
    """Predictor that records the size of every batch it receives."""

    def __init__(self, model):
        self.model = model
        self.batch_sizes = []

    def predict(self, X, **kwargs):
        del kwargs
        self.batch_sizes.append(len(X))
        return np.asarray(self.model.predict(X))


class LinearModel:
    """Tiny deterministic model: y = 2 * f0 - f1."""

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return 2.0 * X[:, 0] - X[:, 1]

    def __repr__(self):
        return "LinearModel()"


def _write_shard(path, X, y, extra_columns=None):
    """Write a shard with a deliberately scrambled column order."""
    data = {
        "ignore_me": np.arange(len(y), dtype=np.float64),
        "f1": X[:, 1],
        "y": y,
        "f0": X[:, 0],
    }
    data.update(extra_columns or {})
    pq.write_table(pa.table(data), path)
    return path


@pytest.fixture
def shards(tmp_path):
    rng = np.random.RandomState(7)
    n = 500
    X = rng.randn(n, 2)
    y = 2.0 * X[:, 0] - X[:, 1] + 0.1 * rng.randn(n)
    # Duplicate a block of samples to check that they are counted twice.
    X = np.concatenate([X, X[:50]])
    y = np.concatenate([y, y[:50]])

    paths = []
    bounds = [0, 180, 360, len(y)]
    for index in range(3):
        path = _write_shard(
            str(tmp_path / f"shard-{index:02d}.parquet"),
            X[bounds[index] : bounds[index + 1]],
            y[bounds[index] : bounds[index + 1]],
        )
        paths.append(path)
    return paths, X, y


def _in_memory_calibrator(X, y):
    model = LinearModel()
    calibrator = BaseCalibrator(
        nonconf_score_func=nonconformity_scores.absolute_difference,
        pred_set_func=constant_interval,
    )
    calibrator.fit(y_pred=model.predict(X), y_true=y)
    return calibrator, model


def test_artifact_scores_match_in_memory_scores(shards, tmp_path):
    shard_paths, X, y = shards
    calibrator, model = _in_memory_calibrator(X, y)

    artifact = build_calibration_artifact(
        path=str(tmp_path / "artifact"),
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(model),
        batch_size=64,
        sort_buffer_scores=40,
        merge_chunk_scores=7,
        max_fan_in=2,
    )
    try:
        assert artifact.n_samples == len(y)
        assert len(artifact) == len(y)
        np.testing.assert_array_equal(
            artifact.get_scores(),
            np.sort(calibrator.get_nonconformity_scores()),
        )
        # Duplicated calibration samples are kept as many times as they appear.
        assert len(np.unique(artifact.get_scores())) < artifact.n_samples
    finally:
        artifact.close()


@pytest.mark.parametrize("alpha", [0.002, 0.01, 0.05, 0.1, 0.5, 0.9, 0.99, 0.999])
def test_artifact_quantile_matches_base_calibrator(shards, tmp_path, alpha):
    shard_paths, X, y = shards
    calibrator, model = _in_memory_calibrator(X, y)

    artifact = build_calibration_artifact(
        path=str(tmp_path / "artifact"),
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(model),
        batch_size=100,
        sort_buffer_scores=64,
    )
    try:
        expected = calibrator.compute_quantile(alpha=alpha)
        np.testing.assert_array_equal(artifact.compute_quantile(alpha=alpha), expected)
        y_pred_lower, y_pred_upper = artifact.calibrate(
            alpha=alpha, y_pred=np.array([0.0, 1.0])
        )
        np.testing.assert_array_equal(y_pred_lower, np.array([0.0, 1.0]) - expected)
        np.testing.assert_array_equal(y_pred_upper, np.array([0.0, 1.0]) + expected)
    finally:
        artifact.close()


def test_artifact_alpha_conventions_match_base_calibrator(shards, tmp_path):
    shard_paths, X, y = shards
    calibrator, model = _in_memory_calibrator(X, y)
    artifact = build_calibration_artifact(
        path=str(tmp_path / "artifact"),
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(model),
    )
    n = len(y)
    try:
        # Smallest admissible alpha.
        np.testing.assert_array_equal(
            artifact.compute_quantile(alpha=1 / (n + 1)),
            calibrator.compute_quantile(alpha=1 / (n + 1)),
        )
        # alpha < 1 / (n + 1) fails with the historical error.
        with pytest.raises(ValueError, match="Alpha is too small"):
            calibrator.compute_quantile(alpha=1 / (n + 2))
        with pytest.raises(ValueError, match="Alpha is too small"):
            artifact.compute_quantile(alpha=1 / (n + 2))
        # alpha >= 1 fails with the historical error (identity correction).
        with pytest.raises(ValueError, match="too large"):
            calibrator.compute_quantile(alpha=1.0, correction=lambda value: value)
        with pytest.raises(ValueError, match="too large"):
            artifact.compute_quantile(alpha=1.0, correction=None)
        with pytest.raises(ValueError, match="alpha must be in"):
            artifact.compute_quantile(alpha=1.0)
    finally:
        artifact.close()


def test_artifact_counts_duplicate_rows_and_bounds_batches(shards, tmp_path):
    shard_paths, _, y = shards
    model = LinearModel()
    predictor = RecordingPredictor(model)

    artifact = build_calibration_artifact(
        path=str(tmp_path / "artifact"),
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=predictor,
        batch_size=32,
    )
    try:
        # The batch size bounds every actual model input.
        assert predictor.batch_sizes
        assert max(predictor.batch_sizes) <= 32
        assert sum(predictor.batch_sizes) == len(y)
        # The duplicated block is counted twice.
        scores = artifact.get_scores()
        assert scores.size == len(y)
    finally:
        artifact.close()


def test_artifact_reopens_without_source_shards(shards, tmp_path):
    shard_paths = shards[0]
    model = LinearModel()
    artifact_path = str(tmp_path / "artifact")
    artifact = build_calibration_artifact(
        path=artifact_path,
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(model),
        save_predictor=True,
    )
    quantile = artifact.compute_quantile(alpha=0.2)
    artifact.close()
    assert artifact.closed

    # The source shards are gone: the artifact must not need them again.
    for path in shard_paths:
        os.remove(path)

    reopened = open_calibration_artifact(artifact_path)
    try:
        np.testing.assert_array_equal(reopened.compute_quantile(alpha=0.2), quantile)
        y_pred, y_lower, y_upper = reopened.predict(
            np.array([[0.0, 0.0], [1.0, 1.0]]), alpha=0.2
        )
        np.testing.assert_allclose(y_pred, [0.0, 1.0])
        np.testing.assert_array_equal(y_lower, y_pred - quantile)
        np.testing.assert_array_equal(y_upper, y_pred + quantile)
    finally:
        reopened.close()


def test_artifact_calibrator_is_base_calibrator_compatible(shards, tmp_path):
    shard_paths, _, y = shards
    artifact = build_calibration_artifact(
        path=str(tmp_path / "artifact"),
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(LinearModel()),
    )
    try:
        calibrator = artifact.to_calibrator()
        assert isinstance(calibrator, BaseCalibrator)
        assert isinstance(calibrator, OutOfCoreCalibrator)
        assert calibrator.get_nonconformity_scores() is None
        assert calibrator.get_norm_weights() is None
        np.testing.assert_array_equal(
            calibrator.compute_quantile(alpha=0.1),
            artifact.compute_quantile(alpha=0.1),
        )
        y_lower, y_upper = calibrator.calibrate(alpha=0.1, y_pred=np.array([0.0, 2.0]))
        np.testing.assert_array_equal(
            y_lower, np.array([0.0, 2.0]) - artifact.compute_quantile(alpha=0.1)
        )
        np.testing.assert_array_equal(
            y_upper, np.array([0.0, 2.0]) + artifact.compute_quantile(alpha=0.1)
        )
        with pytest.raises(RuntimeError, match="cannot be fitted"):
            calibrator.fit(y_true=y, y_pred=y)

        # Deep copies (e.g. CrossValCpAggregator) and pickles still work.
        copied = copy.deepcopy(calibrator)
        assert copied.compute_quantile(alpha=0.1) == calibrator.compute_quantile(
            alpha=0.1
        )
        reloaded = pickle.loads(pickle.dumps(calibrator))
        np.testing.assert_array_equal(
            y_lower, reloaded.calibrate(alpha=0.1, y_pred=np.array([0.0, 2.0]))[0]
        )
    finally:
        artifact.close()


def test_artifact_reports_model_and_schema_bindings(shards, tmp_path):
    shard_paths, _, y = shards
    artifact = build_calibration_artifact(
        path=str(tmp_path / "artifact"),
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(LinearModel()),
    )
    try:
        metadata = artifact.metadata
        assert metadata["n_samples"] == len(y)
        assert metadata["features"] == FEATURES
        assert metadata["target"] == "y"
        assert metadata["input_schema"]["feature_dtypes"] == [
            "double",
            "double",
        ]
        assert metadata["input_schema"]["target_dtype"] == "double"
        assert metadata["model"]["model_class"] == "LinearModel"
        assert metadata["model"]["predictor_class"] == "RecordingPredictor"
        assert len(metadata["shards"]) == len(shard_paths)
        assert artifact.predictor_descriptor["model_repr"] == "LinearModel()"
        # The predictor is not embedded unless requested.
        with pytest.raises(CalibrationArtifactError, match="does not embed"):
            _ = artifact.predictor
    finally:
        artifact.close()


def test_artifact_feature_order_is_the_metadata_order(shards, tmp_path):
    shard_paths = shards[0]
    forward = build_calibration_artifact(
        path=str(tmp_path / "forward"),
        shards=shard_paths,
        features=["f0", "f1"],
        target="y",
        predictor=RecordingPredictor(LinearModel()),
    )
    swapped = build_calibration_artifact(
        path=str(tmp_path / "swapped"),
        shards=shard_paths,
        features=["f1", "f0"],
        target="y",
        predictor=RecordingPredictor(LinearModel()),
    )
    try:
        # Same model, swapped features: the scores must differ.
        assert not np.allclose(forward.get_scores(), swapped.get_scores())
        assert forward.metadata["features"] == ["f0", "f1"]
        assert swapped.metadata["features"] == ["f1", "f0"]
    finally:
        forward.close()
        swapped.close()


def test_artifact_infers_features_from_first_shard(shards, tmp_path):
    shard_paths, _, y = shards
    artifact = build_calibration_artifact(
        path=str(tmp_path / "artifact"),
        shards=shard_paths,
        target="y",
        predictor=RecordingPredictor(LinearModel()),
    )
    try:
        # File order is f1, y, f0 (minus the target).
        assert artifact.features == ("ignore_me", "f1", "f0")
        assert artifact.n_samples == len(y)
    finally:
        artifact.close()


def test_artifact_accepts_directory_and_glob(shards, tmp_path):
    shard_paths, _, y = shards
    directory = str(os.path.dirname(shard_paths[0]))
    artifact = build_calibration_artifact(
        path=str(tmp_path / "from-dir"),
        shards=directory,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(LinearModel()),
    )
    try:
        assert artifact.n_samples == len(y)
    finally:
        artifact.close()

    artifact = build_calibration_artifact(
        path=str(tmp_path / "from-glob"),
        shards=os.path.join(directory, "shard-*.parquet"),
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(LinearModel()),
    )
    try:
        assert artifact.n_samples == len(y)
    finally:
        artifact.close()


def test_incomplete_artifact_is_never_a_usable_calibrator(shards, tmp_path):
    shard_paths, _, y = shards
    artifact_path = tmp_path / "artifact"
    artifact = build_calibration_artifact(
        path=str(artifact_path),
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(LinearModel()),
    )
    artifact.close()

    # Remove the completion marker: the artifact must be rejected.
    marker = artifact_path / COMPLETE_MARKER
    saved_marker = marker.read_bytes()
    marker.unlink()
    with pytest.raises(ArtifactNotReadyError, match="completion marker"):
        open_calibration_artifact(str(artifact_path))
    marker.write_bytes(saved_marker)

    # Tamper with the metadata: the checksum check must reject it.
    metadata_path = artifact_path / METADATA_FILE
    saved_metadata = metadata_path.read_bytes()
    payload = json.loads(saved_metadata)
    payload["n_samples"] = payload["n_samples"] - 1
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactNotReadyError, match="metadata was modified"):
        open_calibration_artifact(str(artifact_path))
    metadata_path.write_bytes(saved_metadata)

    # Truncate the score file: the size check must reject it.
    scores_path = artifact_path / SCORES_FILE
    saved_scores = scores_path.read_bytes()
    scores_path.write_bytes(saved_scores[:-8])
    with pytest.raises(ArtifactNotReadyError, match="corrupted"):
        open_calibration_artifact(str(artifact_path))
    scores_path.write_bytes(saved_scores)

    # The intact artifact still opens.
    reopened = open_calibration_artifact(str(artifact_path))
    try:
        assert reopened.n_samples == len(y)
    finally:
        reopened.close()

    # A directory with no metadata is not an artifact.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ArtifactNotReadyError, match="incomplete"):
        open_calibration_artifact(str(empty))
    with pytest.raises(ArtifactNotReadyError):
        open_calibration_artifact(str(tmp_path / "missing"))


def test_failed_build_leaves_no_usable_artifact(shards, tmp_path):
    shard_paths = shards[0]
    artifact_path = tmp_path / "artifact"

    class FailingPredictor:
        def __init__(self):
            self.calls = 0

        def predict(self, X):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boom")
            return np.zeros(len(X))

    with pytest.raises(RuntimeError, match="boom"):
        build_calibration_artifact(
            path=str(artifact_path),
            shards=shard_paths,
            features=FEATURES,
            target="y",
            predictor=FailingPredictor(),
            batch_size=10,
        )

    assert not artifact_path.exists()
    # No leftover building directory.
    leftovers = [
        name
        for name in os.listdir(tmp_path)
        if name.startswith(f".{artifact_path.name}.building-")
    ]
    assert leftovers == []


def test_failed_overwrite_keeps_previous_artifact(shards, tmp_path):
    shard_paths, _, _ = shards
    artifact_path = str(tmp_path / "artifact")
    original = build_calibration_artifact(
        path=artifact_path,
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(LinearModel()),
    )
    expected = original.compute_quantile(alpha=0.2)
    original.close()

    class FailingPredictor:
        def predict(self, X):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        build_calibration_artifact(
            path=artifact_path,
            shards=shard_paths,
            features=FEATURES,
            target="y",
            predictor=FailingPredictor(),
            overwrite=True,
        )

    # The previous artifact is untouched and still usable.
    reopened = open_calibration_artifact(artifact_path)
    try:
        np.testing.assert_array_equal(reopened.compute_quantile(alpha=0.2), expected)
    finally:
        reopened.close()
    leftovers = [
        name
        for name in os.listdir(tmp_path)
        if name.startswith(f".{os.path.basename(artifact_path)}.")
    ]
    assert leftovers == []


def test_artifact_build_errors(shards, tmp_path):
    shard_paths = shards[0]

    with pytest.raises(ValueError, match="path"):
        build_calibration_artifact(
            shards=shard_paths,
            predictor=RecordingPredictor(LinearModel()),
        )
    with pytest.raises(ValueError, match="predictor"):
        build_calibration_artifact(
            path=str(tmp_path / "a"),
            shards=shard_paths,
            features=FEATURES,
            target="y",
        )
    with pytest.raises(ValueError, match="batch_size"):
        build_calibration_artifact(
            path=str(tmp_path / "a"),
            shards=shard_paths,
            features=FEATURES,
            target="y",
            predictor=RecordingPredictor(LinearModel()),
            batch_size=0,
        )
    with pytest.raises(FileNotFoundError):
        build_calibration_artifact(
            path=str(tmp_path / "a"),
            shards=[str(tmp_path / "missing.parquet")],
            features=FEATURES,
            target="y",
            predictor=RecordingPredictor(LinearModel()),
        )
    with pytest.raises(CalibrationArtifactError, match="missing column"):
        build_calibration_artifact(
            path=str(tmp_path / "a"),
            shards=shard_paths,
            features=["f0", "nope"],
            target="y",
            predictor=RecordingPredictor(LinearModel()),
        )
    with pytest.raises(CalibrationArtifactError, match="missing column"):
        build_calibration_artifact(
            path=str(tmp_path / "a"),
            shards=shard_paths,
            features=FEATURES,
            target="nope",
            predictor=RecordingPredictor(LinearModel()),
        )


def test_artifact_rejects_non_scalar_scores(shards, tmp_path):
    shard_paths = shards[0]

    def vector_scores(y_pred, y_true):
        return np.stack([np.abs(y_pred - y_true)] * 2, axis=-1)

    with pytest.raises(NotImplementedError, match="scalar"):
        build_calibration_artifact(
            path=str(tmp_path / "a"),
            shards=shard_paths,
            features=FEATURES,
            target="y",
            predictor=RecordingPredictor(LinearModel()),
            nonconf_score_func=vector_scores,
        )

    def nan_scores(y_pred, y_true):
        scores = np.abs(y_pred - y_true)
        scores[0] = np.nan
        return scores

    with pytest.raises(CalibrationArtifactError, match="NaN"):
        build_calibration_artifact(
            path=str(tmp_path / "b"),
            shards=shard_paths,
            features=FEATURES,
            target="y",
            predictor=RecordingPredictor(LinearModel()),
            nonconf_score_func=nan_scores,
        )


def test_artifact_rejects_non_numeric_columns(shards, tmp_path):
    shard_paths, _, y = shards
    table = pq.read_table(shard_paths[0])
    data = {name: table.column(name).to_numpy() for name in table.column_names}
    data["f0"] = np.array(["a"] * len(y[:180]))
    pq.write_table(pa.table(data), shard_paths[0])

    with pytest.raises(CalibrationArtifactError, match="non-numeric"):
        build_calibration_artifact(
            path=str(tmp_path / "a"),
            shards=shard_paths,
            features=FEATURES,
            target="y",
            predictor=RecordingPredictor(LinearModel()),
        )


def test_artifact_custom_score_function_is_reused(shards, tmp_path):
    shard_paths, X, y = shards
    artifact = build_calibration_artifact(
        path=str(tmp_path / "artifact"),
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(LinearModel()),
        nonconf_score_func=nonconformity_scores.absolute_difference,
    )
    try:
        expected = np.sort(np.abs(LinearModel().predict(X) - y))
        np.testing.assert_array_equal(artifact.get_scores(), expected)
        assert artifact.nonconf_score_func is (nonconformity_scores.absolute_difference)
    finally:
        artifact.close()


def test_artifact_handles_infinite_scores(shards, tmp_path):
    """Infinite calibration scores are kept, sorted and quantiled exactly."""
    shard_paths, X, y = shards

    def inf_scores(y_pred, y_true):
        scores = np.abs(y_pred - y_true).astype(float)
        scores[np.asarray(y_true) > 1.5] = np.inf
        return scores

    artifact = build_calibration_artifact(
        path=str(tmp_path / "artifact"),
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(LinearModel()),
        nonconf_score_func=inf_scores,
        batch_size=17,
        sort_buffer_scores=11,
        merge_chunk_scores=3,
        max_fan_in=2,
    )
    try:
        reference = np.sort(inf_scores(LinearModel().predict(X), y))
        assert np.isinf(reference).any()
        np.testing.assert_array_equal(artifact.get_scores(), reference)

        # Exact rank with the appended +inf sentinel.
        n = len(reference)
        padded = np.concatenate([reference, [np.inf]])
        for alpha in [1 / (n + 1), 0.05, 0.1, 0.3, 0.5, 0.75, 0.9, 0.999]:
            expected = np.quantile(padded, 1 - alpha, method="inverted_cdf")
            got = artifact.compute_quantile(alpha=alpha)
            assert got == expected or (np.isinf(got) and np.isinf(expected))
    finally:
        artifact.close()


def test_artifact_unsupported_options(shards, tmp_path):
    shard_paths, _, y = shards
    artifact = build_calibration_artifact(
        path=str(tmp_path / "artifact"),
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(LinearModel()),
    )
    try:
        with pytest.raises(NotImplementedError, match="Weighted"):
            artifact.compute_quantile(alpha=0.1, weights=np.ones(len(y)))
        with pytest.raises(NotImplementedError, match="scalar"):
            artifact.compute_quantile(alpha=np.array([0.1, 0.2]))
        with pytest.raises(NotImplementedError, match="scalar"):
            artifact.compute_quantile(alpha=True)
        with pytest.raises(CalibrationArtifactError, match="closed"):
            artifact.close()
            artifact.compute_quantile(alpha=0.1)
    finally:
        artifact.close()


def test_artifact_existing_destination_requires_overwrite(shards, tmp_path):
    shard_paths, _, y = shards
    path = str(tmp_path / "artifact")
    artifact = build_calibration_artifact(
        path=path,
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(LinearModel()),
    )
    artifact.close()

    with pytest.raises(CalibrationArtifactError, match="already exists"):
        build_calibration_artifact(
            path=path,
            shards=shard_paths,
            features=FEATURES,
            target="y",
            predictor=RecordingPredictor(LinearModel()),
        )

    rebuilt = build_calibration_artifact(
        path=path,
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(LinearModel()),
        overwrite=True,
    )
    try:
        assert rebuilt.n_samples == len(y)
    finally:
        rebuilt.close()


def test_artifact_context_manager_and_score_access(shards, tmp_path):
    shard_paths = shards[0]
    path = str(tmp_path / "artifact")
    with build_calibration_artifact(
        path=path,
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(LinearModel()),
    ) as artifact:
        scores = artifact.get_scores()
        assert artifact.score_at(0) == scores[0]
        assert artifact.score_at(-1) == scores[-1]
        chunks = list(artifact.iter_score_chunks(17))
        np.testing.assert_array_equal(np.concatenate(chunks), scores)
        assert all(chunk.size <= 17 for chunk in chunks)
        with pytest.raises(IndexError):
            artifact.score_at(len(scores))
    assert artifact.closed


def test_artifact_matches_split_conformal_predictor_workflow(shards, tmp_path):
    """The artifact scores and intervals match SplitCP on the same data."""
    from deel.puncc.regression import SplitCP

    shard_paths, X, y = shards
    model = LinearModel()
    split_cp = SplitCP(RecordingPredictor(model), train=False)
    split_cp.fit(
        X_calib=X,
        y_calib=y,
    )

    artifact = build_calibration_artifact(
        path=str(tmp_path / "artifact"),
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(model),
        save_predictor=True,
    )
    try:
        alpha = 0.1
        _, expected_lower, expected_upper = split_cp.predict(X[:10], alpha)
        y_pred, y_lower, y_upper = artifact.predict(X[:10], alpha)
        np.testing.assert_allclose(y_pred, model.predict(X[:10]))
        np.testing.assert_allclose(y_lower, expected_lower)
        np.testing.assert_allclose(y_upper, expected_upper)
    finally:
        artifact.close()


def test_artifact_calibrator_works_in_cross_val_aggregator(shards, tmp_path):
    from deel.puncc.api.conformalization import CrossValCpAggregator
    from deel.puncc.api.prediction import BasePredictor

    shard_paths, X, _ = shards
    model = LinearModel()
    artifact = build_calibration_artifact(
        path=str(tmp_path / "artifact"),
        shards=shard_paths,
        features=FEATURES,
        target="y",
        predictor=RecordingPredictor(model),
    )
    try:
        aggregator = CrossValCpAggregator(K=1)
        aggregator.append_predictor(0, BasePredictor(model, is_trained=True))
        aggregator.append_calibrator(0, artifact.to_calibrator())

        y_pred, y_lower, y_upper = aggregator.predict(X[:5], alpha=0.1)
        np.testing.assert_allclose(y_pred, model.predict(X[:5]))
        expected = artifact.calibrate(alpha=0.1, y_pred=model.predict(X[:5]))
        np.testing.assert_allclose(y_lower, expected[0])
        np.testing.assert_allclose(y_upper, expected[1])
    finally:
        artifact.close()
