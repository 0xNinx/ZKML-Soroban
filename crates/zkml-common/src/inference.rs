//! Model inference engine.
//!
//! Executes quantized ML models using fixed-point arithmetic. The same
//! logic runs both natively (for testing) and inside the RISC Zero zkVM
//! guest (for proof generation). Keeping inference in `zkml-common` avoids
//! duplicating the proven path between host and guest.

use crate::activation::relu_vec;
use crate::fixed_point::FixedPoint;
use crate::models::{DecisionTree, DenseLayer, LogisticRegression, Model, TinyMLP, TreeNode};

/// Run inference on a model given a vector of input features.
///
/// Returns the raw fixed-point output value.
pub fn run_inference(model: &Model, inputs: &[FixedPoint]) -> FixedPoint {
    match model {
        Model::DecisionTree(tree) => infer_decision_tree(tree, inputs),
        Model::LogisticRegression(lr) => infer_logistic_regression(lr, inputs),
        Model::TinyMLP(mlp) => infer_tiny_mlp(mlp, inputs),
    }
}

/// Traverse a decision tree and return the leaf value.
///
/// # Threshold semantics
///
/// A sample takes the **left** child when
/// `feature[feature_index].value <= threshold.value` (inclusive / `BRANCH_LEQ`).
/// Values strictly greater than the threshold take the right child. This
/// matches typical ONNX `TreeEnsembleClassifier` `BRANCH_LEQ` behavior and
/// must stay aligned with any future circuit encoding.
fn infer_decision_tree(tree: &DecisionTree, inputs: &[FixedPoint]) -> FixedPoint {
    assert_eq!(
        inputs.len(),
        tree.num_features,
        "input length must match the number of features"
    );

    let mut node_idx = 0;
    loop {
        match &tree.nodes[node_idx] {
            TreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if inputs[*feature_index].value <= threshold.value {
                    node_idx = *left;
                } else {
                    node_idx = *right;
                }
            }
            TreeNode::Leaf { value } => return *value,
        }
    }
}

/// Compute logistic regression output: dot(weights, inputs) + bias.
///
/// Note: The sigmoid activation is omitted because it is not ZK-friendly.
/// Instead, the verifier compares the raw linear output against a threshold.
fn infer_logistic_regression(lr: &LogisticRegression, inputs: &[FixedPoint]) -> FixedPoint {
    assert_eq!(
        inputs.len(),
        lr.weights.len(),
        "input length must match the number of weights"
    );

    let scale = inputs[0].scale;
    let dot: i64 = lr
        .weights
        .iter()
        .zip(inputs.iter())
        .map(|(w, x)| (w.value * x.value) >> scale)
        .sum();

    FixedPoint::from_raw(dot + lr.bias.value, scale)
}

/// Compute one dense layer: `out[j] = sum_i(weight[j,i] * in[i]) + bias[j]`.
///
/// Weights are stored row-major as `weights[j * input_size + i]`.
#[allow(clippy::needless_range_loop)]
fn dense_forward(layer: &DenseLayer, inputs: &[FixedPoint]) -> Vec<FixedPoint> {
    let scale = inputs.first().map(|x| x.scale).unwrap_or(16);
    let mut out = Vec::with_capacity(layer.output_size);
    for j in 0..layer.output_size {
        let mut acc: i64 = layer.biases[j].value;
        for i in 0..layer.input_size {
            let w = layer.weights[j * layer.input_size + i].value;
            acc += (w * inputs[i].value) >> scale;
        }
        out.push(FixedPoint::from_raw(acc, scale));
    }
    out
}

/// Run a forward pass through a tiny MLP using quantized ReLU between layers.
fn infer_tiny_mlp(mlp: &TinyMLP, inputs: &[FixedPoint]) -> FixedPoint {
    let mut activations: Vec<FixedPoint> = inputs.to_vec();
    let last = mlp.layers.len().saturating_sub(1);
    for (idx, layer) in mlp.layers.iter().enumerate() {
        let mut out = dense_forward(layer, &activations);
        if idx != last {
            out = relu_vec(&out);
        }
        activations = out;
    }
    activations
        .first()
        .copied()
        .unwrap_or(FixedPoint::from_raw(0, 16))
}

#[cfg(test)]
mod tests_mlp {
    use super::*;
    use crate::models::{DenseLayer, Model, TinyMLP};

    fn fp(x: f64) -> FixedPoint {
        FixedPoint::quantize(x)
    }

    #[test]
    fn single_layer_identity() {
        // One input, one output, weight 1.0, bias 0.0 -> output equals input.
        let layer = DenseLayer {
            weights: vec![fp(1.0)],
            biases: vec![fp(0.0)],
            input_size: 1,
            output_size: 1,
        };
        let model = Model::TinyMLP(TinyMLP {
            layers: vec![layer],
        });
        let out = run_inference(&model, &[fp(0.7)]);
        assert!((out.dequantize() - 0.7).abs() < 1e-2);
    }
}

/// Return the index of the largest value in a fixed-point vector.
///
/// Used to turn a multi-output MLP layer into a class label without a
/// (ZK-unfriendly) softmax: argmax of the logits equals argmax of softmax.
pub fn argmax(values: &[FixedPoint]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by_key(|(_, v)| v.value)
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests_argmax {
    use super::*;

    #[test]
    fn argmax_picks_highest_logit() {
        let logits = vec![
            FixedPoint::quantize(0.1),
            FixedPoint::quantize(0.9),
            FixedPoint::quantize(0.4),
        ];
        assert_eq!(argmax(&logits), Some(1));
    }
}

/// Run inference for each input row, returning one output per row.
pub fn run_batch(model: &Model, rows: &[Vec<FixedPoint>]) -> Vec<FixedPoint> {
    rows.iter().map(|row| run_inference(model, row)).collect()
}

#[cfg(test)]
mod tests_batch {
    use super::*;
    use crate::models::{LogisticRegression, Model};

    #[test]
    fn batch_matches_single() {
        let model = Model::LogisticRegression(LogisticRegression {
            weights: vec![FixedPoint::quantize(1.0)],
            bias: FixedPoint::quantize(0.0),
        });
        let rows = vec![
            vec![FixedPoint::quantize(0.5)],
            vec![FixedPoint::quantize(0.9)],
        ];
        let batched = run_batch(&model, &rows);
        for (row, out) in rows.iter().zip(batched.iter()) {
            assert_eq!(run_inference(&model, row).value, out.value);
        }
    }
}

/// Validated inference that returns an error instead of panicking on a
/// feature-count mismatch or empty input.
pub fn try_run_inference(
    model: &Model,
    inputs: &[FixedPoint],
) -> Result<FixedPoint, crate::error::ZkmlError> {
    use crate::error::ZkmlError;
    if inputs.is_empty() {
        return Err(ZkmlError::FeatureCountMismatch {
            expected: model.num_features(),
            got: 0,
        });
    }
    let expected = model.num_features();
    if expected != 0 && inputs.len() != expected {
        return Err(ZkmlError::FeatureCountMismatch {
            expected,
            got: inputs.len(),
        });
    }
    Ok(run_inference(model, inputs))
}

#[cfg(test)]
mod tests_validated {
    use super::*;
    use crate::models::{LogisticRegression, Model};

    #[test]
    fn empty_input_is_rejected() {
        let model = Model::LogisticRegression(LogisticRegression {
            weights: vec![FixedPoint::quantize(1.0)],
            bias: FixedPoint::quantize(0.0),
        });
        assert!(try_run_inference(&model, &[]).is_err());
    }
}
