"""CLI for YOLOv26 segmentation finetune experiment.

Commands:
    validate-dataset  Check dataset integrity, optionally clean stale dirs
    train             Train YOLOv26-seg on fungal colony dataset
    infer             Run inference on prepared dataset images
    remote-bootstrap  Clone and setup monorepo on Vast.ai instance
    remote-train      SCP dataset, train remotely, SCP weights back
    remote-infer      Run inference remotely and retrieve outputs
"""

from __future__ import annotations

import argparse
import json
import sys


def _cmd_validate_dataset(args: argparse.Namespace) -> None:
    from src.experiments.yolo_segmentation.prepare import (
        cleanup_stale_datasets,
        validate_yolo_dataset,
    )

    if args.cleanup:
        removed = cleanup_stale_datasets()
        if removed:
            print(f"Removed {len(removed)} stale paths:")
            for r in removed:
                print(f"  {r}")
        else:
            print("No stale paths to remove.")

    report = validate_yolo_dataset()
    print(report.summary())

    if args.split:
        from src.experiments.yolo_segmentation.prepare import (
            create_train_val_split,
            rewrite_dataset_yaml,
        )
        train, val, train_txt, val_txt = create_train_val_split(
            train_ratio=args.train_ratio,
            seed=args.seed,
        )
        print(f"\nSplit: {len(train)} train + {len(val)} val (seed={args.seed})")
        yaml_path = rewrite_dataset_yaml(
            train_list=str(train_txt),
            val_list=str(val_txt),
        )
        print(f"Updated {yaml_path}")

    if not report.valid:
        sys.exit(1)


def _cmd_train(args: argparse.Namespace) -> None:
    from src.experiments.yolo_segmentation.run import run_yolo26_train

    result = run_yolo26_train(
        model_variant=args.model_variant,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2))


def _cmd_infer(args: argparse.Namespace) -> None:
    from src.experiments.yolo_segmentation.run import run_yolo26_inference

    result = run_yolo26_inference(
        weights_path=args.weights,
        data_root=args.data_root or None,
        limit=args.limit or None,
        confidence=args.confidence,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


def _cmd_remote_bootstrap(args: argparse.Namespace) -> None:
    from src.experiments.yolo_segmentation.prepare import bootstrap_remote

    success = bootstrap_remote(
        host=args.host,
        ssh_port=args.ssh_port,
        instance_id=args.instance_id,
        branch=args.branch,
    )
    if success:
        print("Remote bootstrap complete.")
    else:
        print("Remote bootstrap failed.", file=sys.stderr)
        sys.exit(1)


def _cmd_remote_train(args: argparse.Namespace) -> None:
    from src.experiments.yolo_segmentation.prepare import remote_train

    result = remote_train(
        host=args.host,
        ssh_port=args.ssh_port,
        scp_port=args.scp_port,
        model_variant=args.model_variant,
        epochs=args.epochs,
        batch=args.batch,
    )
    print(json.dumps(result, indent=2))


def _cmd_remote_infer(args: argparse.Namespace) -> None:
    from src.experiments.yolo_segmentation.prepare import remote_infer

    result = remote_infer(
        host=args.host,
        ssh_port=args.ssh_port,
        scp_port=args.scp_port,
        weights_path=args.weights,
        limit=args.limit or None,
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="yolo_segmentation experiment CLI")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # validate-dataset
    p_val = sub.add_parser("validate-dataset", help="Validate YOLO dataset integrity")
    p_val.add_argument("--cleanup", action="store_true", help="Remove stale dataset directories")
    p_val.add_argument("--split", action="store_true", help="Create train/val split")
    p_val.add_argument("--train-ratio", type=float, default=0.8)
    p_val.add_argument("--seed", type=int, default=42)

    # train
    p_train = sub.add_parser("train", help="Train YOLOv26-seg model")
    p_train.add_argument("--model-variant", default="n", choices=["n", "s", "m", "l", "x"])
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch", type=int, default=8)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--device", default="0")
    p_train.add_argument("--workers", type=int, default=8)

    # infer
    p_infer = sub.add_parser("infer", help="Run YOLOv26 inference on prepared dataset")
    p_infer.add_argument("--weights", required=True, help="Path to trained .pt file")
    p_infer.add_argument("--data-root", default=None, help="Prepared dataset root")
    p_infer.add_argument("--limit", type=int, default=None)
    p_infer.add_argument("--confidence", type=float, default=0.25)
    p_infer.add_argument("--device", default="0")

    # remote-bootstrap
    p_rb = sub.add_parser("remote-bootstrap", help="Setup monorepo on Vast.ai instance")
    p_rb.add_argument("--host", default="1.208.108.242")
    p_rb.add_argument("--ssh-port", type=int, default=61872)
    p_rb.add_argument("--instance-id", default="36259342")
    p_rb.add_argument("--branch", default=None, help="Branch to checkout (default: current)")

    # remote-train
    p_rt = sub.add_parser("remote-train", help="SCP dataset, train remotely, retrieve weights")
    p_rt.add_argument("--host", default="1.208.108.242")
    p_rt.add_argument("--ssh-port", type=int, default=61872)
    p_rt.add_argument("--scp-port", type=int, default=61872)
    p_rt.add_argument("--model-variant", default="n", choices=["n", "s", "m", "l", "x"])
    p_rt.add_argument("--epochs", type=int, default=30)
    p_rt.add_argument("--batch", type=int, default=8)
    p_rt.add_argument("--workers", type=int, default=2)

    # remote-infer
    p_ri = sub.add_parser("remote-infer", help="Run inference remotely and retrieve outputs")
    p_ri.add_argument("--host", default="1.208.108.242")
    p_ri.add_argument("--ssh-port", type=int, default=61872)
    p_ri.add_argument("--scp-port", type=int, default=61872)
    p_ri.add_argument("--weights", required=True)
    p_ri.add_argument("--limit", type=int, default=None)

    # Legacy compatibility
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--description", default="")

    args = parser.parse_args()

    if not args.command:
        if args.run_id:
            from src.experiments.yolo_segmentation.run import ExperimentParams, run as _run
            params = ExperimentParams(
                run_id=args.run_id,
                output_root=args.output_root or "/tmp",
                description=args.description,
            )
            try:
                result = _run(params)
            except Exception as exc:
                print(f"Experiment failure: {exc}", file=sys.stderr)
                sys.exit(1)
            print(json.dumps({
                "f1_score": result.f1_score,
                "strategy_name": result.strategy_name,
                "artifact_paths": result.artifact_paths,
                "run_id": result.run_id,
            }, indent=2))
            return
        parser.print_help()
        sys.exit(1)

    try:
        {
            "validate-dataset": _cmd_validate_dataset,
            "train": _cmd_train,
            "infer": _cmd_infer,
            "remote-bootstrap": _cmd_remote_bootstrap,
            "remote-train": _cmd_remote_train,
            "remote-infer": _cmd_remote_infer,
        }[args.command](args)
    except ImportError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"Command failure: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
