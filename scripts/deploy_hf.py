"""Publish the trained weights (model repo) and the web demo (Docker Space) to Hugging Face.

One-time login first:  .venv/bin/hf auth login     (token with "write" access)

  .venv/bin/python -m scripts.deploy_hf --user <hf-username> --dry-run   # show exactly what would upload
  .venv/bin/python -m scripts.deploy_hf --user <hf-username>

Uploads use an explicit allow-list, so personal photos, datasets, results and anything else in the
working tree can never be published by accident.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ["analyzer.pt", "lut.pt", "restorer.pt"]
SPACE_FILES = ["photofix", "server", "web", "models", "requirements.txt", ".dockerignore"]
NEVER = ("Original Photos", "data", "results", "checkpoints", ".venv", "__pycache__")


def stage_space(model_repo: str, space_url: str) -> Path:
    """Copy exactly the files the Space needs into a temp dir, with repo names filled in."""
    stage = Path(tempfile.mkdtemp(prefix="photofix-space-"))
    for name in SPACE_FILES:
        src = ROOT / name
        if src.is_dir():
            shutil.copytree(src, stage / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        else:
            shutil.copy2(src, stage / name)
    dockerfile = (ROOT / "Dockerfile").read_text()
    (stage / "Dockerfile").write_text(dockerfile.replace("ARG MODEL_REPO=Samdany/photofix-models",
                                                         f"ARG MODEL_REPO={model_repo}"))
    readme = (ROOT / "deploy" / "space_README.md").read_text()
    (stage / "README.md").write_text(readme.replace("{MODEL_REPO}", model_repo).replace("{SPACE_URL}", space_url))
    leaked = [p for p in stage.rglob("*") if any(part in NEVER for part in p.relative_to(stage).parts)]
    if leaked:
        raise SystemExit(f"Refusing to deploy, unexpected files staged: {leaked[:5]}")
    return stage


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="Hugging Face username or organization")
    ap.add_argument("--model-repo", default="photofix-models")
    ap.add_argument("--space", default="PhotoFix")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--demo-url", help="where the demo is hosted, for the model card (default: the Space)")
    ap.add_argument("--weights-only", action="store_true", help="publish the model repo, skip the Space")
    args = ap.parse_args()

    model_repo = f"{args.user}/{args.model_repo}"
    space_repo = f"{args.user}/{args.space}"
    space_url = f"https://huggingface.co/spaces/{space_repo}"
    missing = [w for w in WEIGHTS if not (ROOT / "checkpoints" / w).exists()]
    if missing:
        raise SystemExit(f"Missing checkpoints: {missing}. Train them first (see README).")

    stage = stage_space(model_repo, space_url)
    card = (ROOT / "deploy" / "model_card.md").read_text().replace("{MODEL_REPO}", model_repo).replace("{DEMO_URL}", args.demo_url or space_url)
    files = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
    size = sum((stage / f).stat().st_size for f in files) / 1e6
    print(f"Model repo {model_repo}: {', '.join(WEIGHTS)} + README.md (model card)")
    print(f"Space {space_repo}: {len(files)} files, {size:.1f} MB")
    for f in files:
        print("   ", f)
    if args.dry_run:
        print("\nDry run: nothing uploaded.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"\nLogged in as: {api.whoami()['name']}")
    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    for w in WEIGHTS:
        api.upload_file(path_or_fileobj=str(ROOT / "checkpoints" / w), path_in_repo=w, repo_id=model_repo)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=model_repo)
    print(f"Weights published: https://huggingface.co/{model_repo}")
    if args.weights_only:
        return

    api.create_repo(space_repo, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=str(stage), repo_id=space_repo, repo_type="space",
                      commit_message="Deploy PhotoFix demo", delete_patterns=["*"])
    print(f"Space deployed (first build takes ~5-10 min): {space_url}")


if __name__ == "__main__":
    main()
