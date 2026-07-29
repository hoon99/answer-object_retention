# Answer-Object Retention

Data, prompts, normalized model rankings, and evaluation code for:

> **Found but Not Held: Answer-Object Retention in 3D Referring Expressions**

The release supports the fixed-inventory analysis reported in the paper. It
contains the final family-unique evaluation core: 30 ScanNet v2 validation
scenes, 411 condition sets, 1,834 descriptions, 1,574 full expressions, and
2,256 proper-containment full-expression--broader-description pairs.

## Contents

```text
data/
  scenes.jsonl
  condition_sets.jsonl
  descriptions.jsonl
  pairs.jsonl
prompts/
  main_natural.jsonl
  independent_natural.jsonl
  generic_head.jsonl
  headless.jsonl
rankings/
  <prompt_set>/<system>.jsonl
evaluation/
  evaluate.py
manifest.json
SHA256SUMS
```

The five released systems are 3DRS, Ross3D, Video-3D LLM, 3D-VisTA, and
PQ3D. Each ranking file contains one complete normalized ordering of the
aligned candidate inventory per released description. Rankings are sorted
from highest to lowest score; remaining ties are resolved by ascending object
ID.

## Data Schema

`data/scenes.jsonl` stores the aligned candidate inventory for each scene,
including object IDs and axis-aligned boxes.

`data/condition_sets.jsonl` stores each condition family, its textual value,
and the object-ID set that satisfies it.

`data/descriptions.jsonl` stores the condition combination and object-ID
denotation for every full or broader description.

`data/pairs.jsonl` stores the evaluated proper-containment links. For each
pair, `full_denotation` is a proper subset of `broader_denotation`, and
`removed_condition_type` identifies the omitted condition family.

## Prompt Sets

- `main_natural`: primary natural realization used for the main results.
- `independent_natural`: independently generated natural realization used for
  surface-form robustness.
- `generic_head`: canonical generic-head realization used for the
  category-omission wording control.
- `headless`: canonical headless realization used for the same control.

The generic-head and headless files cover the 329 broader descriptions used
in that wording analysis. The two natural files cover all 1,834 released
descriptions.

## Recompute Core Results

The evaluator uses only the Python standard library:

```bash
python3 evaluation/evaluate.py \
  --root . \
  --output results/core_metrics.json
```

By default, it computes 2,000 scene-clustered bootstrap replicates and reports:

- endpoint set accuracy;
- retained endpoint accuracy;
- all-broader retention;
- pair-weighted answer recall and pair-pass rate;
- exact broader-description recovery and chance-corrected recall;
- omitted-condition profiles;
- cutoff sensitivity;
- independent-natural and category-wording robustness.

For a quick integrity check, use fewer bootstrap replicates:

```bash
python3 evaluation/evaluate.py \
  --root . \
  --bootstrap-iterations 20 \
  --output results/smoke_test.json
```

Validate file counts, joins, proper containment, and complete candidate
rankings with:

```bash
python3 evaluation/validate_release.py
```

## Source Resources

The release does not redistribute ScanNet point clouds, meshes, or images.
Scene IDs, aligned object IDs, boxes, and class metadata are derived from the
ScanNet and Sr3D/ReferIt3D resources used in the paper. Users must obtain those
source resources separately and follow their respective terms of use.

The canonical ScanNet scene IDs retain the `_00` suffix in the files because
it is required for reliable joins with the source resources.
