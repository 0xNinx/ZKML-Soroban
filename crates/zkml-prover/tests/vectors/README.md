# Golden Test Vectors

This directory contains golden-vector test cases for the ZKML inference engine. These vectors serve as the canonical test suite for inference correctness and are reused across multiple testing contexts:

- **Native Rust tests** (`golden_vectors.rs`) - validates the inference engine implementation
- **RISC Zero guest tests** (future) - validates the zkVM guest program produces identical results
- **Phase 2 circuit tests** (future) - validates the ZK circuit computes the same values

## Format

Each vector is a JSON file with the following structure:

```json
{
  "description": "Human-readable description of the test vector",
  "model": {
    "type": "DecisionTree|LogisticRegression",
    ...model-specific fields...
  },
  "test_cases": [
    {
      "description": "Description of this specific test case",
      "inputs": [
        {"value": <i64>, "scale": <u32>}
      ],
      "expected_output": {"value": <i64>, "scale": <u32>},
      "expected_error": "<optional error type string>"
    }
  ]
}
```

### Fixed-Point Representation

All numeric values use Q16.16 fixed-point format:
- `value`: The raw integer value (real_value × 2^16)
- `scale`: Always 16 for this project

Example: The floating-point value 0.5 is represented as `{"value": 32768, "scale": 16}` since 0.5 × 65536 = 32768.

### Model Types

#### Decision Tree

```json
{
  "type": "DecisionTree",
  "num_features": <number of input features>,
  "nodes": [
    {
      "type": "Split",
      "feature_index": <index>,
      "threshold": {"value": <i64>, "scale": 16},
      "left": <child index>,
      "right": <child index>
    },
    {
      "type": "Leaf",
      "value": {"value": <i64>, "scale": 16}
    }
  ]
}
```

#### Logistic Regression

```json
{
  "type": "LogisticRegression",
  "weights": [
    {"value": <i64>, "scale": 16},
    ...
  ],
  "bias": {"value": <i64>, "scale": 16}
}
```

## Test Coverage

The current vector suite covers:

### Decision Tree Cases
- `decision_tree_depth1.json` - Depth-1 stump (single split), both branches exercised
- `decision_tree_depth3.json` - Depth-3 tree, every leaf reachable
- `decision_tree_boundary.json` - Boundary behavior (input exactly at threshold)
- `decision_tree_degenerate.json` - Single leaf root (no splits)

### Logistic Regression Cases
- `logistic_regression_positive.json` - Positive weights, positive score
- `logistic_regression_negative.json` - Negative weights and negative features
- `logistic_regression_zero.json` - Zero score (cancellation)

### Error Paths
- `error_feature_mismatch.json` - Feature vector length mismatch
- `error_out_of_range_child.json` - Tree with out-of-range child index

## Computing Expected Values

The `compute_expected.py` script verifies hand-computed expected values using the same arithmetic as the Rust implementation:

```bash
python3 compute_expected.py
```

This script implements the Q16.16 fixed-point arithmetic used by the inference engine, including:
- Quantization/dequantization
- Dot product with right-shift scaling
- Logistic regression output computation

## Threshold Semantics

Decision tree splits use **less-than-or-equal** (`<=`) semantics:
- If `input[feature] <= threshold`, traverse to the **left** child
- If `input[feature] > threshold`, traverse to the **right** child

This matches ONNX's `BRANCH_LEQ` operator and is documented in `inference.rs`. The boundary behavior is tested in `decision_tree_boundary.json`.

## Adding New Vectors

When adding new test vectors:

1. Create a new JSON file following the format above
2. Compute expected values by hand or using `compute_expected.py`
3. Store expected outputs as raw Q16.16 integers (no floating-point ambiguity)
4. Add the file name to the `generate_vector_tests!` macro in `golden_vectors.rs`
5. Run `cargo test -p zkml-prover` to verify

## Reuse in Other Contexts

These vectors are designed to be language-agnostic. To reuse them:

1. Parse the JSON file
2. Convert the model structure to your target language's representation
3. Convert inputs from Q16.16 to your fixed-point format
4. Run inference
5. Compare outputs against the expected Q16.16 values

The raw integer format ensures there's no floating-point ambiguity across different implementations.
