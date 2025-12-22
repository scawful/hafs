# Training Snapshot Schema

This schema documents the JSON snapshots consumed by the viz app. Fields are
intentionally conservative and aligned to existing emitters. Optional fields may
be added without breaking current consumers.

## Common Metadata (Optional)
Add to any snapshot file when available.
- `generated_at` (string, ISO 8601): when the snapshot was generated.
- `sequence` (integer): monotonically increasing sequence number.
- `source` (string): emitter identity (hostname, node name, or service).

## quality_feedback.json
Top-level object.
- `generator_stats` (object)
  - keys: generator name
  - values: object
    - `samples_generated` (int)
    - `samples_accepted` (int)
    - `samples_rejected` (int)
    - `avg_quality_score` (float, 0..1)
    - `rejection_reasons` (object: string -> int)
- `rejection_history` (array)
  - each entry: object
    - `domain` (string)
    - `scores` (object: metric -> float)

## active_learning.json
Top-level object.
- `regions` (array)
  - each entry: object
    - `sample_count` (int)
    - `domain` (string)
    - `avg_quality` (float, 0..1)

## training_feedback.json
Top-level object.
- `training_runs` (object)
  - keys: run_id
  - values: object
    - `model_name` (string)
    - `base_model` (string)
    - `dataset_path` (string)
    - `samples_count` (int)
    - `start_time` (string, ISO 8601)
    - `end_time` (string, ISO 8601)
    - `final_loss` (float)
    - `eval_metrics` (object: metric -> float)
    - `domain_distribution` (object: domain -> int)
    - `notes` (string)
- `domain_effectiveness` (object: domain -> float)
- `quality_threshold_effectiveness` (object: threshold -> float)
  - JSON object keys are strings; callers should parse to float.
