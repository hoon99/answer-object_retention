#!/usr/bin/env python3
"""Recompute the core answer-object retention results.

This script uses only the Python standard library. It reports the primary
endpoint and retention metrics, omitted-condition profiles, cutoff
sensitivity, and natural-realization robustness for the five released systems.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from collections import defaultdict
from pathlib import Path


MODELS = ["3drs", "ross3d", "video_3d_llm", "3d_vista", "pq3d"]
MODEL_LABELS = {
    "3drs": "3DRS",
    "ross3d": "Ross3D",
    "video_3d_llm": "Video-3D LLM",
    "3d_vista": "3D-VisTA",
    "pq3d": "PQ3D",
}
BOOTSTRAP_MODEL_LABELS = {
    **MODEL_LABELS,
    "video_3d_llm": "Video-3D-LLM",
}
CONDITIONS = [
    "category",
    "attribute",
    "anchor_relation",
    "scene_region",
    "viewpoint",
]
SEED = 20260719


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def quantile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    position = q * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return (
        values[lower] * (upper - position)
        + values[upper] * (position - lower)
    )


def clustered_interval(records, value_key, label, iterations, predicate=None):
    predicate = predicate or (lambda row: True)
    by_scene = defaultdict(list)
    for record in records:
        by_scene[record["scene_id"]].append(record)
    scenes = sorted(by_scene)
    if not scenes:
        return [float("nan"), float("nan")]
    rng = random.Random(SEED + zlib.crc32(label.encode("utf-8")))
    estimates = []
    for _ in range(iterations):
        sample = []
        for scene in rng.choices(scenes, k=len(scenes)):
            sample.extend(by_scene[scene])
        estimates.append(
            mean(row[value_key] for row in sample if predicate(row))
        )
    return [quantile(estimates, 0.025), quantile(estimates, 0.975)]


def summarize(records, value_key, label, iterations, predicate=None):
    predicate = predicate or (lambda row: True)
    selected = [row for row in records if predicate(row)]
    return {
        "point": mean(row[value_key] for row in selected),
        "ci95": clustered_interval(
            records, value_key, label, iterations, predicate
        ),
        "n": len(selected),
        "scenes": len({row["scene_id"] for row in selected}),
    }


def ranked_ids(row):
    return [int(item["object_id"]) for item in row["object_rankings"]]


def top_set(row, size):
    return set(ranked_ids(row)[:size])


def load_rankings(root, prompt_set, model):
    path = root / "rankings" / prompt_set / f"{model}.jsonl"
    rows = {row["semantic_id"]: row for row in read_jsonl(path)}
    return rows


def build_records(pairs, rankings, cutoff_ratio=1.0):
    by_full = defaultdict(list)
    for pair in pairs:
        by_full[pair["full_semantic_id"]].append(pair)

    expressions = []
    pair_records = []
    for full_id, full_pairs in by_full.items():
        full_gold = set(map(int, full_pairs[0]["full_denotation"]))
        endpoint_prediction = top_set(rankings[full_id], len(full_gold))
        endpoint = float(endpoint_prediction == full_gold)
        pair_passes = []
        for pair in full_pairs:
            broader_gold = set(map(int, pair["broader_denotation"]))
            budget = min(
                int(pair["candidate_count"]),
                max(1, math.ceil(cutoff_ratio * len(broader_gold))),
            )
            prediction = top_set(rankings[pair["broader_semantic_id"]], budget)
            answer_recall = len(full_gold & prediction) / len(full_gold)
            pair_pass = float(full_gold <= prediction)
            broader_exact = float(
                budget == len(broader_gold) and prediction == broader_gold
            )
            random_expectation = budget / int(pair["candidate_count"])
            chance_corrected = (
                (answer_recall - random_expectation) / (1 - random_expectation)
                if random_expectation < 1
                else float(answer_recall == 1)
            )
            pair_passes.append(pair_pass)
            pair_records.append(
                {
                    "scene_id": pair["scene_id"],
                    "full_semantic_id": full_id,
                    "link_id": pair["link_id"],
                    "removed_condition_type": pair["removed_condition_type"],
                    "endpoint": endpoint,
                    "answer_recall": answer_recall,
                    "pair_pass": pair_pass,
                    "broader_exact": broader_exact,
                    "chance_corrected_answer_recall": chance_corrected,
                }
            )

        all_broader = float(all(value == 1 for value in pair_passes))
        expressions.append(
            {
                "scene_id": full_pairs[0]["scene_id"],
                "full_semantic_id": full_id,
                "endpoint": endpoint,
                "retained_endpoint": endpoint * all_broader,
                "all_broader": all_broader,
                "broader_description_count": len(full_pairs),
            }
        )
    return expressions, pair_records


def summarize_run(expressions, pairs, label, iterations):
    endpoint_only = lambda row: row["endpoint"] == 1
    return {
        "endpoint_set_accuracy": summarize(
            expressions, "endpoint", f"{label}:endpoint", iterations
        ),
        "retained_endpoint_accuracy": summarize(
            expressions,
            "retained_endpoint",
            f"{label}:joint",
            iterations,
        ),
        "all_broader_retention": summarize(
            expressions,
            "all_broader",
            f"{label}:strict_ep",
            iterations,
            endpoint_only,
        ),
        "pair_weighted_answer_recall": summarize(
            pairs,
            "answer_recall",
            f"{label}:link_ep",
            iterations,
            endpoint_only,
        ),
        "pair_pass_rate": summarize(
            pairs,
            "pair_pass",
            f"{label}:pair_pass_ep",
            iterations,
            endpoint_only,
        ),
        "broader_exact_recovery": summarize(
            pairs,
            "broader_exact",
            f"{label}:exact_broader_ep",
            iterations,
            endpoint_only,
        ),
        "chance_corrected_answer_recall": summarize(
            pairs,
            "chance_corrected_answer_recall",
            f"{label}:chance-corrected",
            iterations,
            endpoint_only,
        ),
    }


def condition_profile(pair_records, label, iterations):
    endpoint_pairs = [row for row in pair_records if row["endpoint"] == 1]
    output = {}
    for condition in CONDITIONS:
        subset = [
            row
            for row in endpoint_pairs
            if row["removed_condition_type"] == condition
        ]
        output[condition] = {
            "answer_recall": summarize(
                subset,
                "answer_recall",
                f"{label}:{condition}:answer",
                iterations,
            ),
            "pair_pass_rate": summarize(
                subset,
                "pair_pass",
                f"{label}:{condition}:pass",
                iterations,
            ),
        }
    return output


def category_wording_control(root, pairs, main_rankings, model, iterations):
    variants = {}
    for prompt_set in ["generic_head", "headless"]:
        variant_rankings = load_rankings(root, prompt_set, model)
        eligible = [
            pair
            for pair in pairs
            if pair["removed_condition_type"] == "category"
            and pair["broader_semantic_id"] in variant_rankings
        ]
        records = []
        for pair in eligible:
            full_gold = set(map(int, pair["full_denotation"]))
            endpoint = top_set(main_rankings[pair["full_semantic_id"]], len(full_gold))
            if endpoint != full_gold:
                continue
            broader_gold = set(map(int, pair["broader_denotation"]))
            prediction = top_set(
                variant_rankings[pair["broader_semantic_id"]],
                len(broader_gold),
            )
            records.append(
                {
                    "scene_id": pair["scene_id"],
                    "answer_recall": len(full_gold & prediction) / len(full_gold),
                    "pair_pass": float(full_gold <= prediction),
                }
            )
        variants[prompt_set] = {
            "answer_recall": summarize(
                records,
                "answer_recall",
                f"wording:{model}:{prompt_set}:answer",
                iterations,
            ),
            "pair_pass_rate": summarize(
                records,
                "pair_pass",
                f"wording:{model}:{prompt_set}:pass",
                iterations,
            ),
        }
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/core_metrics.json"))
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    args = parser.parse_args()

    pairs = list(read_jsonl(args.root / "data" / "pairs.jsonl"))
    result = {
        "metadata": {
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_cluster": "ScanNet scene",
            "seed": SEED,
            "primary_cutoff": "cardinality of the broader denotation",
        },
        "main": {},
        "condition_profiles": {},
        "cutoff_sensitivity": {},
        "independent_natural": {},
        "category_wording": {},
    }

    for model in MODELS:
        main_rankings = load_rankings(args.root, "main_natural", model)
        expressions, pair_records = build_records(pairs, main_rankings)
        result["main"][MODEL_LABELS[model]] = summarize_run(
            expressions,
            pair_records,
            f"main:{BOOTSTRAP_MODEL_LABELS[model]}",
            args.bootstrap_iterations,
        )
        result["condition_profiles"][MODEL_LABELS[model]] = condition_profile(
            pair_records,
            f"condition:{model}",
            args.bootstrap_iterations,
        )

        result["cutoff_sensitivity"][MODEL_LABELS[model]] = {}
        for ratio in [0.5, 1.0, 2.0, 3.0]:
            ratio_expressions, ratio_pairs = build_records(
                pairs, main_rankings, cutoff_ratio=ratio
            )
            summary = summarize_run(
                ratio_expressions,
                ratio_pairs,
                f"cutoff:{model}:{ratio}",
                args.bootstrap_iterations,
            )
            result["cutoff_sensitivity"][MODEL_LABELS[model]][str(ratio)] = {
                "all_broader_retention": summary["all_broader_retention"],
                "pair_weighted_answer_recall": summary[
                    "pair_weighted_answer_recall"
                ],
            }

        independent_rankings = load_rankings(
            args.root, "independent_natural", model
        )
        independent_expressions, independent_pairs = build_records(
            pairs, independent_rankings
        )
        result["independent_natural"][MODEL_LABELS[model]] = summarize_run(
            independent_expressions,
            independent_pairs,
            f"independent:{model}",
            args.bootstrap_iterations,
        )
        result["category_wording"][MODEL_LABELS[model]] = (
            category_wording_control(
                args.root,
                pairs,
                main_rankings,
                model,
                args.bootstrap_iterations,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
