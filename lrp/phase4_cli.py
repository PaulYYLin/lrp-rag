"""CLI for the locked HotpotQA evaluation."""
from pathlib import Path


def register(commands):
    ranking = Path("artifacts/phase4-retrieval")
    answers = Path("artifacts/phase4-answers")
    controls = Path("artifacts/phase4-controls")
    for name in ("calibrate256", "retrieve", "controls", "answer", "report", "timing"):
        cmd = commands.add_parser(f"phase4-{name}", help=f"HotpotQA {name}")
        cmd.add_argument("--prepared", type=Path, default=Path("artifacts/marco" if name == "calibrate256" else "artifacts/hotpot"))
        if name == "retrieve":
            group = cmd.add_mutually_exclusive_group()
            group.add_argument("--calibration256", type=Path, help="Run only the supplemental learned-256 pair")
            group.add_argument("--pca256", action="store_true", help="Run only the supplemental PCA-256 exact search")
        if name in ("calibrate256", "retrieve", "timing"):
            cmd.add_argument("--lock", type=Path, default=Path("artifacts/phase2-lock.json"))
            cmd.add_argument("--cache", type=Path, default=Path("data/embeddings"))
            cmd.add_argument("--model-dir", type=Path, default=Path("data/models/contriever"))
            cmd.add_argument("--device", default="mps")
        if name in ("answer", "report", "timing"):
            cmd.add_argument("--retrieval", type=Path, default=ranking)
        if name in ("report", "timing"):
            cmd.add_argument("--retrieval-pca256", type=Path)
        if name in ("answer", "controls"):
            cmd.add_argument("--generator", type=Path, default=Path("data/models/qwen2.5-0.5b-instruct"))
            cmd.add_argument("--device", choices=("cpu", "cuda", "mps"), default="mps")
            cmd.add_argument("--generation-cache", type=Path, default=Path("data/phase4-generation"))
        if name == "timing":
            cmd.add_argument("--retrieval256", type=Path, default=Path("artifacts/phase4-retrieval-256"))
            cmd.add_argument("--report", type=Path, default=Path("artifacts/phase4-report"))
        if name == "report":
            cmd.add_argument("--retrieval256", type=Path)
            cmd.add_argument("--answers256", type=Path)
            cmd.add_argument("--answers-pca256", type=Path)
            cmd.add_argument("--controls", type=Path, default=controls)
            cmd.add_argument("--answers", type=Path, default=answers)
            cmd.add_argument("--scorer", type=Path, default=Path("data/official/hotpot_evaluate_v1.py"))
            cmd.add_argument("--nltk-data", type=Path, default=Path("data/nltk"))
        default = dict(timing=Path("artifacts/phase4-timing"), calibrate256=Path("artifacts/phase4-dev-256"), retrieve=ranking, answer=answers, controls=controls, report=Path("artifacts/phase4-report"))
        cmd.add_argument("--out", type=Path, default=default[name])
        cmd.set_defaults(action=dispatch)


def dispatch(args):
    if args.command == "phase4-calibrate256":
        from .phase4_256 import calibrate
        calibrate(args)
    elif args.command == "phase4-retrieve":
        from .phase4 import retrieve
        retrieve(args)
    elif args.command == "phase4-timing":
        from .phase4_timing import measure
        measure(args)
    elif args.command == "phase4-report":
        from .phase4_report import report
        report(args)
    else:
        from .phase4_qa import answer
        answer(args, controls=args.command == "phase4-controls")
