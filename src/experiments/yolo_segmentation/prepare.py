"""YOLOv26 finetune preparation: cleanup, dataset validation, split, remote helpers."""

from __future__ import annotations

import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.config import (
    WORKSPACE_ROOT,
    YOLO_DATASET_DIR,
    YOLO_WEIGHTS_DIR,
    YOLO_RESULTS_DIR,
    YOLO_CLASS_NAMES,
    PREPARED_DATASET_DIR,
)


@dataclass
class DatasetReport:
    total_images: int = 0
    total_labels: int = 0
    unpaired_images: list[str] = field(default_factory=list)
    unpaired_labels: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    valid: bool = False

    def summary(self) -> str:
        lines = [
            f"Images: {self.total_images}",
            f"Labels: {self.total_labels}",
            f"Classes: {self.class_names}",
            f"Unpaired images: {len(self.unpaired_images)}",
            f"Unpaired labels: {len(self.unpaired_labels)}",
            f"Valid: {self.valid}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prepare Roboflow-format dataset yaml for YOLOv26 training
# ---------------------------------------------------------------------------

def prepare_roboflow_dataset_yaml(dataset_dir: Path | None = None) -> tuple[int, int]:
    root = dataset_dir or YOLO_DATASET_DIR
    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text()) or {}
    data["path"] = str(root.resolve())

    train_dir = root / "train" / "images"
    val_dir = root / "test" / "images"
    data["train"] = "train/images"
    data["val"] = "test/images"

    if train_dir.exists():
        data["train"] = str(train_dir.resolve())
    if val_dir.exists():
        data["val"] = str(val_dir.resolve())

    yaml_path.write_text(yaml.dump(data, default_flow_style=False))

    train_count = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.jpeg"))) + len(list(train_dir.glob("*.png")))
    val_count = len(list(val_dir.glob("*.jpg"))) + len(list(val_dir.glob("*.jpeg"))) + len(list(val_dir.glob("*.png")))

    return train_count, val_count


# ---------------------------------------------------------------------------
# T005 — cleanup stale dataset directories
# ---------------------------------------------------------------------------

STALE_DIRS = [
    WORKSPACE_ROOT / "Dataset" / "full_image",
    WORKSPACE_ROOT / "Dataset" / "segmented_image",
]
STALE_FILES = [
    WORKSPACE_ROOT / "Dataset" / "full_image_metadata.json",
    WORKSPACE_ROOT / "Dataset" / "segmented_image_metadata.json",
]


def cleanup_stale_datasets() -> list[str]:
    removed: list[str] = []
    for d in STALE_DIRS:
        if d.exists():
            shutil.rmtree(d)
            removed.append(str(d.relative_to(WORKSPACE_ROOT)))
    for f in STALE_FILES:
        if f.exists():
            f.unlink()
            removed.append(str(f.relative_to(WORKSPACE_ROOT)))
    return removed


# ---------------------------------------------------------------------------
# T006 — validate roboflow species dataset
# ---------------------------------------------------------------------------

def validate_yolo_dataset(dataset_dir: Path | None = None) -> DatasetReport:
    root = dataset_dir or YOLO_DATASET_DIR
    images_dir = root / "images"
    labels_dir = root / "labels"
    classes_file = root / "classes.txt"
    yaml_file = root / "dataset.yaml"

    report = DatasetReport()

    if not root.exists():
        return report

    img_exts = {".jpg", ".jpeg", ".png"}
    image_stems: set[str] = set()
    if images_dir.exists():
        for img in images_dir.iterdir():
            if img.is_file() and img.suffix.lower() in img_exts:
                image_stems.add(img.stem)
    report.total_images = len(image_stems)

    label_stems: set[str] = set()
    if labels_dir.exists():
        for lbl in labels_dir.iterdir():
            if lbl.is_file() and lbl.suffix == ".txt":
                label_stems.add(lbl.stem)
    report.total_labels = len(label_stems)

    report.unpaired_images = sorted(image_stems - label_stems)
    report.unpaired_labels = sorted(label_stems - image_stems)

    if classes_file.exists():
        report.class_names = [
            line.strip()
            for line in classes_file.read_text().splitlines()
            if line.strip()
        ]

    report.valid = (
        report.total_images > 0
        and report.total_images == report.total_labels
        and len(report.unpaired_images) == 0
        and len(report.unpaired_labels) == 0
        and yaml_file.exists()
        and classes_file.exists()
    )

    return report


# ---------------------------------------------------------------------------
# T007 — rewrite dataset.yaml paths at runtime
# ---------------------------------------------------------------------------

def rewrite_dataset_yaml(
    dataset_dir: Path | None = None,
    train_list: str | None = None,
    val_list: str | None = None,
) -> Path:
    root = dataset_dir or YOLO_DATASET_DIR
    yaml_path = root / "dataset.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {yaml_path}")

    data: dict[str, Any] = {}
    if yaml_path.exists():
        data = yaml.safe_load(yaml_path.read_text()) or {}

    data["path"] = str(root.resolve())
    data["train"] = train_list or str((root / "images").resolve())
    data["val"] = val_list or str((root / "images").resolve())

    if "names" not in data:
        data["names"] = YOLO_CLASS_NAMES

    yaml_path.write_text(yaml.dump(data, default_flow_style=False))
    return yaml_path


# ---------------------------------------------------------------------------
# T008 — 80/20 train/val split
# ---------------------------------------------------------------------------

def create_train_val_split(
    dataset_dir: Path | None = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str], Path, Path]:
    root = dataset_dir or YOLO_DATASET_DIR
    images_dir = root / "images"
    img_exts = {".jpg", ".jpeg", ".png"}

    image_names = sorted(
        img.name
        for img in images_dir.iterdir()
        if img.is_file() and img.suffix.lower() in img_exts
    )

    if not image_names:
        raise FileNotFoundError(f"No images found in {images_dir}")

    random.seed(seed)
    shuffled = image_names[:]
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_images = sorted(shuffled[:split_idx])
    val_images = sorted(shuffled[split_idx:])

    train_txt = root / "train.txt"
    val_txt = root / "val.txt"
    train_txt.write_text("\n".join(str(images_dir / name) for name in train_images))
    val_txt.write_text("\n".join(str(images_dir / name) for name in val_images))

    return train_images, val_images, train_txt, val_txt


# ---------------------------------------------------------------------------
# T038/T039 — SSH & SCP helpers
# ---------------------------------------------------------------------------

def _ssh_cmd(host: str, ssh_port: int, command: str) -> list[str]:
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        "-p", str(ssh_port),
        f"root@{host}",
        command,
    ]


def _scp_cmd(
    host: str, scp_port: int,
    src: str, dst: str,
    recursive: bool = False,
) -> list[str]:
    cmd = ["scp", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10", "-P", str(scp_port)]
    if recursive:
        cmd.append("-r")
    cmd.extend([src, f"root@{host}:{dst}"])
    return cmd


def _scp_pull_cmd(
    host: str, scp_port: int,
    src: str, dst: str,
    recursive: bool = False,
) -> list[str]:
    cmd = ["scp", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10", "-P", str(scp_port)]
    if recursive:
        cmd.append("-r")
    cmd.extend([f"root@{host}:{src}", dst])
    return cmd


def ssh_run(
    host: str,
    command: str,
    ssh_port: int = 61872,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    cmd = _ssh_cmd(host, ssh_port, command)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def scp_transfer(
    host: str,
    local_path: str,
    remote_path: str,
    scp_port: int = 61888,
    direction: str = "push",
    recursive: bool = True,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    if direction == "push":
        cmd = _scp_cmd(host, scp_port, local_path, remote_path, recursive=recursive)
    else:
        cmd = _scp_pull_cmd(host, scp_port, remote_path, local_path, recursive=recursive)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------------------
# T024 — nohup wrapper for remote training
# ---------------------------------------------------------------------------

def remote_nohup_train(
    host: str,
    ssh_port: int,
    model_variant: str = "n",
    epochs: int = 30,
    batch: int = 8,
    workers: int = 2,
    remote_workspace: str = "/root/mycoai",
) -> int | None:
    cmd = (
        f"cd {remote_workspace}/repos/fungal-cv-qdrant && "
        f"MYCOAI_ROOT={remote_workspace} "
        f"nohup uv run python "
        f"-m src.experiments.yolo_segmentation.cli train "
        f"--model-variant {model_variant} --epochs {epochs} --batch {batch} "
        f"--workers {workers} "
        f"> /tmp/yolo26_train.log 2>&1 & echo $!"
    )
    result = ssh_run(host, cmd, ssh_port=ssh_port, timeout=30)
    if result.returncode == 0 and result.stdout.strip():
        pid = int(result.stdout.strip())
        print(f"Training launched on remote, PID={pid}")
        return pid
    print(f"Failed to launch remote training: {result.stderr}", flush=True)
    return None


# ---------------------------------------------------------------------------
# T040 — remote-bootstrap
# ---------------------------------------------------------------------------

def bootstrap_remote(
    host: str,
    ssh_port: int = 61872,
    instance_id: str = "36259342",
    branch: str | None = None,
    repo_url: str = "https://github.com/datamonsterr/fungal-cv-qdrant.git",
    remote_workspace: str = "/root/mycoai",
) -> bool:

    branch = branch or "006-yolo26-seg-finetune"

    submodule_dir = f"{remote_workspace}/repos/fungal-cv-qdrant"

    steps = [
        f"mkdir -p {remote_workspace}/repos && rm -rf {submodule_dir} && git clone {repo_url} {submodule_dir}",
        f"cd {submodule_dir} && git checkout {branch}",
        f"mkdir -p {remote_workspace}/Dataset {remote_workspace}/weights/yolo26 {remote_workspace}/results",
        "pip install uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; true",
        f"cd {submodule_dir} && MYCOAI_ROOT={remote_workspace} uv sync",
    ]

    for i, step in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {step[:80]}...")
        result = ssh_run(host, step, ssh_port=ssh_port, timeout=900)
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[:200]}")
            return False
        print("  OK")
    return True


# ---------------------------------------------------------------------------
# T041 — remote-train
# ---------------------------------------------------------------------------

def remote_train(
    host: str,
    ssh_port: int = 61872,
    scp_port: int = 61872,
    model_variant: str = "n",
    epochs: int = 30,
    batch: int = 8,
    remote_workspace: str = "/root/mycoai",
) -> dict[str, Any]:
    local_dataset = str(YOLO_DATASET_DIR)
    remote_dataset = f"{remote_workspace}/Dataset/"

    print(f"Pushing dataset {local_dataset} to {host}:{remote_dataset}...")
    result = scp_transfer(host, local_dataset, remote_dataset, scp_port=scp_port, direction="push")
    if result.returncode != 0:
        return {"status": "failed", "reason": "dataset transfer failed", "stderr": result.stderr[:500]}

    pid = remote_nohup_train(
        host=host,
        ssh_port=ssh_port,
        model_variant=model_variant,
        epochs=epochs,
        batch=batch,
        remote_workspace=remote_workspace,
    )
    if pid is None:
        return {"status": "failed", "reason": "training launch failed"}

    # Poll for completion
    remote_weights = f"{remote_workspace}/weights/yolo26/"
    remote_results = f"{remote_workspace}/results/yolo26_finetune/"
    local_weights = str(YOLO_WEIGHTS_DIR)
    local_results = str(YOLO_RESULTS_DIR)

    print(f"Training launched (PID={pid}). Waiting for completion...")
    max_wait = epochs * 120  # rough upper bound in seconds
    waited = 0
    while waited < max_wait:
        time.sleep(60)
        waited += 60
        check = ssh_run(host, f"ls {remote_weights}/yolo26n-seg_species_best.pt 2>/dev/null && echo DONE || echo WAITING", ssh_port=ssh_port, timeout=15)
        if "DONE" in check.stdout:
            break

    print("Downloading weights...")
    Path(local_weights).mkdir(parents=True, exist_ok=True)
    scp_transfer(host, f"{remote_weights}*", local_weights, scp_port=scp_port, direction="pull")

    print("Downloading results...")
    Path(local_results).mkdir(parents=True, exist_ok=True)
    scp_transfer(host, f"{remote_results}/*", local_results, scp_port=scp_port, direction="pull")

    return {
        "status": "completed",
        "pid": pid,
        "local_weights": local_weights,
        "local_results": local_results,
    }


# ---------------------------------------------------------------------------
# T042 — remote-infer
# ---------------------------------------------------------------------------

def remote_infer(
    host: str,
    ssh_port: int = 61872,
    scp_port: int = 61872,
    weights_path: str = "",
    limit: int | None = None,
    remote_workspace: str = "/root/mycoai",
) -> dict[str, Any]:
    local_prepared = str(PREPARED_DATASET_DIR)
    remote_prepared = f"{remote_workspace}/Dataset/prepared/"

    print("Pushing prepared dataset...")
    result = scp_transfer(host, local_prepared, remote_prepared, scp_port=scp_port, direction="push")
    if result.returncode != 0:
        return {"status": "failed", "reason": "dataset transfer failed"}

    limit_flag = f"--limit {limit}" if limit else ""
    weights_name = Path(weights_path).name if weights_path else "yolo26n-seg_species_best.pt"
    remote_weights = f"{remote_workspace}/weights/yolo26/{weights_name}"

    cmd = (
        f"cd {remote_workspace}/repos/fungal-cv-qdrant && "
        f"MYCOAI_ROOT={remote_workspace} "
        f"uv run python "
        f"-m src.experiments.yolo_segmentation.cli infer "
        f"--weights {remote_weights} --data-root {remote_workspace}/Dataset/prepared {limit_flag}"
    )
    print("Running remote inference...")
    result = ssh_run(host, cmd, ssh_port=ssh_port, timeout=7200)
    if result.returncode != 0:
        return {"status": "failed", "reason": "inference failed", "stderr": result.stderr[:500]}

    return {"status": "completed", "output": result.stdout[-500:]}
