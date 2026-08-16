import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--dev-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)

    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    # token_root will be the directory containing the three token directories.
    token_root = args.output.parent.resolve()

    sources = [
        ("train", args.train_dir.resolve()),
        ("valid", args.dev_dir.resolve()),
        ("test", args.test_dir.resolve()),
    ]

    merged_rows = []
    counts = {}

    for split, source_dir in sources:

        index_path = source_dir / "tokens.jsonl"

        if not index_path.is_file():
            raise FileNotFoundError(
                f"tokens.jsonl not found: {index_path}"
            )

        count = 0

        with index_path.open("r", encoding="utf-8") as f:

            for line_number, line in enumerate(f, start=1):

                if not line.strip():
                    continue

                row = json.loads(line)

                if "token_path" not in row:
                    raise ValueError(
                        f"{index_path}:{line_number} "
                        "does not contain token_path"
                    )

                # Locate the original .pt file.
                old_token_path = (
                    source_dir / row["token_path"]
                ).resolve()

                if not old_token_path.is_file():
                    raise FileNotFoundError(
                        f"Token file not found: {old_token_path}"
                    )

                # Rewrite path relative to the common token root.
                try:
                    new_token_path = old_token_path.relative_to(token_root)
                except ValueError as exc:
                    raise ValueError(
                        f"{old_token_path} is not inside "
                        f"common token root {token_root}"
                    ) from exc

                row["token_path"] = new_token_path.as_posix()

                # Explicitly standardize split names expected by train_probe.py.
                row["split"] = split

                merged_rows.append(row)
                count += 1

        counts[split] = count

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        for row in merged_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )

    print("Merge completed.")
    print()

    for split, count in counts.items():
        print(f"{split}: {count}")

    print(f"total: {len(merged_rows)}")
    print()
    print(f"output: {args.output.resolve()}")
    print(f"token root: {token_root}")


if __name__ == "__main__":
    main()