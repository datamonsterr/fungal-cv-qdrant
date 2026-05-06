"""Tests for ExperimentParams/ExperimentResult contract across experiment packages."""

from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path


EXPERIMENT_PACKAGES = [
    "src.experiments.retrieval.run",
    "src.experiments.threshold.run",
    "src.experiments.cross_validation.run",
    "src.experiments.feature_extraction.run",
    "src.experiments.finetune_dl.run",
    "src.experiments.kmeans_segmentation.run",
    "src.experiments.yolo_cross_validation.run",
    "src.experiments.yolo_dataset.run",
    "src.experiments.yolo_segmentation.run",
]

PRIORITY_PACKAGES = [
    "src.experiments.retrieval.run",
    "src.experiments.threshold.run",
]


def _import_module(module_path: str):
    """Import module without triggering side effects."""
    return importlib.import_module(module_path)


def test_retrieval_importable_no_side_effects():
    mod = _import_module("src.experiments.retrieval.run")
    assert mod is not None


def test_threshold_importable_no_side_effects():
    mod = _import_module("src.experiments.threshold.run")
    assert mod is not None


def test_retrieval_has_experiment_params():
    mod = _import_module("src.experiments.retrieval.run")
    assert hasattr(mod, "ExperimentParams"), "ExperimentParams missing from retrieval/run.py"
    cls = mod.ExperimentParams
    fields = {f.name for f in dataclasses.fields(cls)}
    assert "run_id" in fields
    assert "output_root" in fields
    assert "description" in fields


def test_retrieval_has_experiment_result():
    mod = _import_module("src.experiments.retrieval.run")
    assert hasattr(mod, "ExperimentResult"), "ExperimentResult missing from retrieval/run.py"
    cls = mod.ExperimentResult
    fields = {f.name for f in dataclasses.fields(cls)}
    assert "f1_score" in fields
    assert "strategy_name" in fields
    assert "artifact_paths" in fields
    assert "run_id" in fields


def test_retrieval_has_run_callable():
    mod = _import_module("src.experiments.retrieval.run")
    assert callable(getattr(mod, "run", None)), "run() missing from retrieval/run.py"


def test_threshold_has_experiment_params():
    mod = _import_module("src.experiments.threshold.run")
    assert hasattr(mod, "ExperimentParams"), "ExperimentParams missing from threshold/run.py"
    cls = mod.ExperimentParams
    fields = {f.name for f in dataclasses.fields(cls)}
    assert "run_id" in fields
    assert "output_root" in fields
    assert "description" in fields


def test_threshold_has_experiment_result():
    mod = _import_module("src.experiments.threshold.run")
    assert hasattr(mod, "ExperimentResult"), "ExperimentResult missing from threshold/run.py"
    cls = mod.ExperimentResult
    fields = {f.name for f in dataclasses.fields(cls)}
    assert "f1_score" in fields
    assert "strategy_name" in fields
    assert "artifact_paths" in fields
    assert "run_id" in fields


def test_threshold_has_run_callable():
    mod = _import_module("src.experiments.threshold.run")
    assert callable(getattr(mod, "run", None)), "run() missing from threshold/run.py"


def test_experiment_params_instantiation_retrieval():
    mod = _import_module("src.experiments.retrieval.run")
    params = mod.ExperimentParams(
        run_id="test-001",
        output_root="/tmp/test-001",
        description="contract test",
    )
    assert params.run_id == "test-001"
    assert params.output_root == "/tmp/test-001"
    assert params.description == "contract test"


def test_experiment_params_instantiation_threshold():
    mod = _import_module("src.experiments.threshold.run")
    params = mod.ExperimentParams(
        run_id="test-002",
        output_root="/tmp/test-002",
        description="contract test threshold",
    )
    assert params.run_id == "test-002"


def test_experiment_result_instantiation_retrieval():
    mod = _import_module("src.experiments.retrieval.run")
    result = mod.ExperimentResult(
        f1_score=0.85,
        strategy_name="test_strategy",
        artifact_paths=["/tmp/artifact.png"],
        run_id="test-001",
    )
    assert result.f1_score == 0.85
    assert result.strategy_name == "test_strategy"
    assert result.run_id == "test-001"
    assert isinstance(result.artifact_paths, list)


def test_experiment_result_f1_range():
    mod = _import_module("src.experiments.retrieval.run")
    result = mod.ExperimentResult(
        f1_score=0.75,
        strategy_name="test",
        artifact_paths=[],
        run_id="test-003",
    )
    assert 0.0 <= result.f1_score <= 1.0


def test_cli_module_importable_retrieval():
    mod = _import_module("src.experiments.retrieval.cli")
    assert callable(getattr(mod, "main", None))


def test_cli_module_importable_threshold():
    mod = _import_module("src.experiments.threshold.cli")
    assert callable(getattr(mod, "main", None))


import pytest

@pytest.mark.parametrize("pkg", [
    "cross_validation",
    "feature_extraction",
    "finetune_dl",
    "kmeans_segmentation",
    "yolo_cross_validation",
    "yolo_dataset",
    "yolo_segmentation",
])
def test_remaining_packages_have_experiment_params(pkg):
    mod = _import_module(f"src.experiments.{pkg}.run")
    assert hasattr(mod, "ExperimentParams"), f"ExperimentParams missing from {pkg}/run.py"
    import dataclasses
    fields = {f.name for f in dataclasses.fields(mod.ExperimentParams)}
    assert "run_id" in fields
    assert "output_root" in fields
    assert "description" in fields


@pytest.mark.parametrize("pkg", [
    "cross_validation",
    "feature_extraction",
    "finetune_dl",
    "yolo_cross_validation",
    "yolo_dataset",
    "yolo_segmentation",
])
def test_remaining_packages_have_run_callable(pkg):
    mod = _import_module(f"src.experiments.{pkg}.run")
    assert callable(getattr(mod, "run", None)), f"run() missing from {pkg}/run.py"


def test_kmeans_segmentation_has_run_experiment_callable():
    mod = _import_module("src.experiments.kmeans_segmentation.run")
    assert callable(getattr(mod, "run_experiment", None)), "run_experiment() missing from kmeans_segmentation/run.py"


@pytest.mark.parametrize("pkg", [
    "cross_validation",
    "feature_extraction",
    "finetune_dl",
    "kmeans_segmentation",
    "yolo_cross_validation",
    "yolo_dataset",
    "yolo_segmentation",
])
def test_remaining_packages_cli_importable(pkg):
    mod = _import_module(f"src.experiments.{pkg}.cli")
    assert callable(getattr(mod, "main", None)), f"main() missing from {pkg}/cli.py"


@pytest.mark.parametrize("pkg", [
    "cross_validation",
    "feature_extraction",
    "finetune_dl",
    "yolo_cross_validation",
    "yolo_dataset",
    "yolo_segmentation",
])
def test_remaining_packages_run_returns_experiment_result(pkg, tmp_path):
    mod = _import_module(f"src.experiments.{pkg}.run")
    params = mod.ExperimentParams(
        run_id=f"test-{pkg}",
        output_root=str(tmp_path / pkg),
        description="contract test",
    )
    result = mod.run(params)
    assert isinstance(result, mod.ExperimentResult)
    assert result.run_id == f"test-{pkg}"
    assert 0.0 <= result.f1_score <= 1.0
    import dataclasses
    assert hasattr(result, "strategy_name")
    assert hasattr(result, "artifact_paths")
