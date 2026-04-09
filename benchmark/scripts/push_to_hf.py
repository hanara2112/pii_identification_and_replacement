import argparse
from pathlib import Path

from huggingface_hub import HfApi

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Push SAHA-AL Benchmark to Hugging Face")
    parser.add_argument("--repo_id", type=str, default="huggingbahl21/saha-al", help="Hugging Face Hub Repo ID")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing train/validation/test jsonl files (default: benchmark/data)",
    )
    parser.add_argument(
        "--token", type=str, default=None, help="Hugging Face Token (optional if already logged in via huggingface-cli)"
    )
    parser.add_argument(
        "--readme-only",
        action="store_true",
        help="Only upload README.md (skip dataset push and benchmark_eval.py)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else SCRIPT_DIR / "data"
    api = HfApi(token=args.token)

    if not args.readme_only:
        from datasets import load_dataset

        print(f"Loading dataset from {data_dir}...")
        dataset = load_dataset(
            "json",
            data_files={
                "train": str(data_dir / "train.jsonl"),
                "validation": str(data_dir / "validation.jsonl"),
                "test": str(data_dir / "test.jsonl"),
            },
        )

        print(f"Pushing dataset to Hugging Face hub ({args.repo_id})...")
        dataset.push_to_hub(args.repo_id, token=args.token)

        print("Uploading README.md and benchmark_eval.py...")
        eval_script = SCRIPT_DIR / "benchmark_eval.py"
        api.upload_file(
            path_or_fileobj=str(eval_script),
            path_in_repo="benchmark_eval.py",
            repo_id=args.repo_id,
            repo_type="dataset",
        )
    else:
        print(f"Uploading README.md to {args.repo_id} ...")

    readme = SCRIPT_DIR / "README.md"
    if not readme.is_file():
        raise FileNotFoundError(f"Missing {readme}")

    api.upload_file(
        path_or_fileobj=str(readme),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    print("Success! Your benchmark is now live.")

if __name__ == "__main__":
    main()
