"""Prepare identical train, validation and official test CSVs for all models."""

import argparse
import csv
import hashlib
import io
import json
import tempfile
import unicodedata
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
TRAIN = "offenseval-tr-training-v1.tsv"
TEST = "offenseval-tr-testset-v1.tsv"
GOLD = "offenseval-tr-labela-v1.tsv"
LABELS = {"NOT": 0, "OFF": 1}
SEED = 42


def read_release(archive, source_dir):
    members = (TRAIN, TEST, GOLD)
    if source_dir:
        paths = {}
        for name in members:
            matches = list(source_dir.rglob(name))
            if len(matches) != 1:
                raise ValueError(f"Expected one {name} under {source_dir}, found {len(matches)}")
            paths[name] = matches[0].read_bytes()
        return paths
    if archive:
        contents = archive.read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            contents = response.read()
    with zipfile.ZipFile(io.BytesIO(contents)) as bundle:
        output = {}
        for name in members:
            candidates = [p for p in bundle.namelist() if p.rsplit("/", 1)[-1] == name]
            if len(candidates) != 1:
                raise ValueError(f"Expected one archive member named {name}")
            output[name] = bundle.read(candidates[0])
    return output


def parse_release(files):
    def rows(name, delimiter):
        reader = io.StringIO(files[name].decode("utf-8-sig"), newline="")
        return list(csv.reader(reader, delimiter=delimiter, quoting=csv.QUOTE_NONE))

    train = rows(TRAIN, "\t")
    test = rows(TEST, "\t")
    gold = rows(GOLD, ",")
    if train.pop(0) != ["id", "tweet", "subtask_a"] or test.pop(0) != ["id", "tweet"]:
        raise ValueError("Unexpected OffensEval-TR headers.")
    if (len(train), len(test), len(gold)) != (31756, 3528, 3528):
        raise ValueError("Expected the complete 2020 release: 31,756 train and 3,528 test rows.")
    gold_by_id = {key: LABELS[label] for key, label in gold}
    train_ids = {r[0] for r in train}
    test_ids = {r[0] for r in test}
    if len(train_ids) != len(train) or len(test_ids) != len(test) or train_ids & test_ids:
        raise ValueError("Duplicate or overlapping example identifiers.")
    if test_ids != gold_by_id.keys():
        raise ValueError("Test labels and messages have different identifiers.")
    training = [(key, text, LABELS[label]) for key, text, label in train]
    testing = [(key, text, gold_by_id[key]) for key, text in test]
    if any(not text.strip() for _, text, _ in training + testing):
        raise ValueError("Empty message in release.")
    return training, testing


def text_key(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def prepare(files, output):
    training, test = parse_release(files)
    test_keys = {text_key(text) for _, text, _ in test}
    seen = set()
    groups = defaultdict(list)
    duplicate_count = overlap_count = 0
    for key, text, label in training:
        normalized = text_key(text)
        if normalized in test_keys:
            overlap_count += 1
            continue
        if normalized in seen:
            duplicate_count += 1
            continue
        seen.add(normalized)
        groups[label].append((key, text, label))

    # Use stable hash ordering to reserve 10% of each training class for validation.
    def order(row):
        return hashlib.sha256(f"{SEED}:{row[0]}".encode()).digest()

    train, validation = [], []
    for group in groups.values():
        group.sort(key=order)
        boundary = round(len(group) * 0.1)
        validation.extend(group[:boundary])
        train.extend(group[boundary:])
    splits = {"train": sorted(train, key=order), "validation": sorted(validation, key=order),
              "test": test}
    summary = {
        "release": "OffensEval-TR 2020",
        "source_url": URL,
        "source_file_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "source_training_rows": len(training),
        "duplicate_training_rows_removed": duplicate_count,
        "training_rows_matching_test_removed": overlap_count,
        "split_seed": SEED,
        "validation_fraction_per_class": 0.1,
        "label_map": LABELS,
        "splits": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temp_dir:
        temp = Path(temp_dir)
        for name, rows in splits.items():
            path = temp / f"{name}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(("text", "label"))
                writer.writerows((text, label) for _, text, label in rows)
            counts = Counter(label for _, _, label in rows)
            summary["splits"][name] = {"rows": len(rows), "NOT": counts[0], "OFF": counts[1],
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (temp / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for path in sorted(temp.iterdir(), key=lambda p: p.suffix == ".json"):
            path.replace(output / path.name)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="Local copy of the official ZIP")
    source.add_argument("--source-dir", type=Path, help="Directory containing the three release TSVs")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    try:
        report = prepare(read_release(args.archive, args.source_dir), args.output)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Data preparation failed: {error}\nFor offline use, pass --archive PATH or --source-dir DIR.\n")
    for name, counts in report["splits"].items():
        print(f"{name:10} {counts['rows']:6,} rows  NOT {counts['NOT']:,}  OFF {counts['OFF']:,}")
