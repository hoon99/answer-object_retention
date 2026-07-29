#!/usr/bin/env python3
"""Validate joins, counts, and complete rankings in the public release."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "manifest.json").read_text())
    scenes = read_jsonl(root / "data" / "scenes.jsonl")
    conditions = read_jsonl(root / "data" / "condition_sets.jsonl")
    descriptions = read_jsonl(root / "data" / "descriptions.jsonl")
    pairs = read_jsonl(root / "data" / "pairs.jsonl")

    expected = manifest["counts"]
    observed = {
        "scenes": len(scenes),
        "condition_sets": len(conditions),
        "descriptions": len(descriptions),
        "full_expressions": sum("full" in row["roles"] for row in descriptions),
        "pairs": len(pairs),
        "generic_head_descriptions": manifest["prompt_counts"]["generic_head"],
        "headless_descriptions": manifest["prompt_counts"]["headless"],
    }
    assert observed == expected, (observed, expected)

    scene_ids = {row["scene_id"] for row in scenes}
    condition_ids = {row["constraint_id"] for row in conditions}
    description_by_id = {row["semantic_id"]: row for row in descriptions}
    assert len(description_by_id) == len(descriptions)

    for row in descriptions:
        assert row["scene_id"] in scene_ids
        assert set(row["constraint_ids"]) <= condition_ids

    for pair in pairs:
        assert pair["full_semantic_id"] in description_by_id
        assert pair["broader_semantic_id"] in description_by_id
        assert set(pair["full_denotation"]) < set(pair["broader_denotation"])

    for prompt_set, expected_count in manifest["prompt_counts"].items():
        prompts = read_jsonl(root / "prompts" / f"{prompt_set}.jsonl")
        prompt_ids = {row["semantic_id"] for row in prompts}
        assert len(prompts) == expected_count
        assert len(prompt_ids) == expected_count
        for model, ranking_count in manifest["ranking_counts"][prompt_set].items():
            rankings = read_jsonl(
                root / "rankings" / prompt_set / f"{model}.jsonl"
            )
            ranking_ids = {row["semantic_id"] for row in rankings}
            assert len(rankings) == ranking_count == expected_count
            assert ranking_ids == prompt_ids
            for row in rankings:
                ordered_ids = [
                    int(item["object_id"]) for item in row["object_rankings"]
                ]
                assert ordered_ids == list(dict.fromkeys(ordered_ids))
                assert set(ordered_ids) == set(row["candidate_object_ids"])

    print(json.dumps(observed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
