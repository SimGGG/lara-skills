"""Command-line interface: toy data, engine build, evaluation and recommendations."""

import argparse
import json
import sys
from pathlib import Path

from ._version import __version__
from .app import STAGES, build, load_app_config, stage_config, workspace_paths
from .models import VARIANTS


def _add_common(parser, config_required=True):
    parser.add_argument("--config", type=Path, required=config_required, help="application config (lara_app.json)")
    parser.add_argument("--root", type=Path, help="workspace root; defaults to the config file's directory")
    parser.add_argument("--variant", choices=sorted(VARIANTS), help="model variant; defaults to the config value")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="override the encoder and training device")


def _load(args):
    app = load_app_config(args.config)
    root = (args.root or args.config.parent).resolve()
    return app, root


def cmd_toy_data(args):
    from .toy import write_toy_inputs
    manifest = write_toy_inputs(args.output, args.seed)
    print(json.dumps({"output": str(Path(args.output).resolve()), "tables": {k: v["rows"] for k, v in manifest["tables"].items()},
                      "app_config": manifest["app_config"]}, indent=2))


def cmd_build(args):
    app, root = _load(args)
    outputs = build(root, app, args.variant, args.device, args.stage)
    print(json.dumps({"root": str(root), "variant": args.variant or app["model"]["variant"], "outputs": outputs}, indent=2))


def cmd_evaluate(args):
    from .evaluate import evaluate_split
    app, root = _load(args)
    variant = args.variant or app["model"]["variant"]
    config = stage_config(app, "evaluate", variant, args.device)
    config.update({"split": args.split, "output_dir": f"{workspace_paths(app, variant)['evaluation']}_{args.split}"})
    if args.output:
        config["output_dir"] = args.output
    evaluate_split(root, config)


def cmd_recommend(args):
    from .recommender import RecommendationEngine
    app, root = _load(args)
    variant = args.variant or app["model"]["variant"]
    config = stage_config(app, "evaluate", variant, args.device)
    engine = RecommendationEngine(root, config["model_dir"], model=config["model"], device=config["device"])
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8-sig")
    rows = engine.recommend_table(text, args.top_k)
    print(json.dumps({"variant": variant, "model": config["model"], "top_k": args.top_k, "recommendations": rows},
                     ensure_ascii=False, indent=2))


def cmd_quickstart(args):
    from .evaluate import evaluate_split
    from .recommender import RecommendationEngine
    from .toy import SAMPLE_DESCRIPTION, write_toy_inputs
    workspace = Path(args.workspace).resolve()
    if not (workspace / "toy_manifest.json").exists():
        workspace.mkdir(parents=True, exist_ok=True)
        write_toy_inputs(workspace, args.seed)
    app = load_app_config(workspace / "lara_app.json")
    variant = args.variant or app["model"]["variant"]
    build(workspace, app, variant, args.device)
    config = stage_config(app, "evaluate", variant, args.device)
    evaluation_dir = f"{workspace_paths(app, variant)['evaluation']}_validation"
    if (workspace / evaluation_dir / "manifest.json").exists():
        result = json.loads((workspace / evaluation_dir / "summary.json").read_text(encoding="utf-8"))
    else:
        result = evaluate_split(workspace, {**config, "split": "validation", "output_dir": evaluation_dir})
    engine = RecommendationEngine(workspace, config["model_dir"], model=config["model"], device=config["device"])
    rows = engine.recommend_table(SAMPLE_DESCRIPTION, 5)
    print(json.dumps({"workspace": str(workspace), "variant": variant,
                      "validation_known_positive_recall_at_5": result["known_positive_recall_at_5"],
                      "validation_known_positive_hit_at_5": result["known_positive_hit_at_5"],
                      "sample_description": SAMPLE_DESCRIPTION, "recommendations": rows,
                      "note": "Synthetic toy data with a hashed n-gram encoder: this checks the pipeline, not recommendation quality."},
                     ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lara-skills", description="LARA technology-to-skill recommendation framework")
    parser.add_argument("--version", action="version", version=f"lara-skills {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    toy = sub.add_parser("toy-data", help="write the synthetic toy inputs and an application config")
    toy.add_argument("--output", type=Path, required=True)
    toy.add_argument("--seed", type=int, default=20261011)
    toy.set_defaults(func=cmd_toy_data)

    builder = sub.add_parser("build", help="run every stage from raw inputs to a trained engine")
    _add_common(builder)
    builder.add_argument("--stage", action="append", choices=STAGES, help="run only the given stage(s), in pipeline order")
    builder.set_defaults(func=cmd_build)

    evaluation = sub.add_parser("evaluate", help="score a labeled split with the trained engine")
    _add_common(evaluation)
    evaluation.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluation.add_argument("--output", help="output directory relative to the workspace (default: artifacts/evaluation/...)")
    evaluation.set_defaults(func=cmd_evaluate)

    recommend = sub.add_parser("recommend", help="recommend skills for a technology description")
    _add_common(recommend)
    source = recommend.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    recommend.add_argument("--top-k", type=int, default=5)
    recommend.set_defaults(func=cmd_recommend)

    quick = sub.add_parser("quickstart", help="generate toy data, build the engine, evaluate and recommend")
    quick.add_argument("--workspace", type=Path, default=Path("lara_toy_workspace"))
    quick.add_argument("--variant", choices=sorted(VARIANTS))
    quick.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    quick.add_argument("--seed", type=int, default=20261011)
    quick.set_defaults(func=cmd_quickstart)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
