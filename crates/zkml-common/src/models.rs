//! Model representation types.
//!
//! Defines the intermediate representations for supported ML model
//! architectures. These structures are produced by the ONNX importer
//! in the prover crate and consumed by both the inference engine and
//! the circuit generator.

use serde::{Deserialize, Serialize};

use crate::fixed_point::FixedPoint;

// ---------------------------------------------------------------------------
// Decision Tree
// ---------------------------------------------------------------------------

/// A single node in a binary decision tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeNode {
    /// Internal split node.
    Split {
        /// Index of the feature to evaluate.
        feature_index: usize,
        /// Threshold value (fixed-point).
        threshold: FixedPoint,
        /// Index of the left child node.
        left: usize,
        /// Index of the right child node.
        right: usize,
    },
    /// Terminal leaf node.
    Leaf {
        /// The predicted class label or regression value.
        value: FixedPoint,
    },
}

/// A complete decision tree represented as a flat node vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    /// Flat array of tree nodes; index 0 is the root.
    pub nodes: Vec<TreeNode>,
    /// Number of input features expected.
    pub num_features: usize,
}

// ---------------------------------------------------------------------------
// Logistic Regression
// ---------------------------------------------------------------------------

/// A binary logistic regression model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegression {
    /// Weight vector (one per feature), in fixed-point.
    pub weights: Vec<FixedPoint>,
    /// Bias term, in fixed-point.
    pub bias: FixedPoint,
}

// ---------------------------------------------------------------------------
// Tiny MLP
// ---------------------------------------------------------------------------

/// A single dense (fully connected) layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseLayer {
    /// Weight matrix stored in row-major order: `weights[i * in + j]`.
    pub weights: Vec<FixedPoint>,
    /// Bias vector, one entry per output neuron.
    pub biases: Vec<FixedPoint>,
    /// Number of input neurons.
    pub input_size: usize,
    /// Number of output neurons.
    pub output_size: usize,
}

/// A small multi-layer perceptron with quantized ReLU activations.
///
/// # Activation convention
///
/// Inference applies quantized ReLU (`max(0, x)`) after every dense layer
/// except the last. The final layer emits raw linear scores; argmax or
/// thresholding is the caller's concern (same rationale as omitting the
/// sigmoid for [`LogisticRegression`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinyMLP {
    /// Ordered list of dense layers.
    pub layers: Vec<DenseLayer>,
}

impl TinyMLP {
    /// Validate layer weight/bias lengths and that consecutive layers chain.
    ///
    /// Checks, before any arithmetic:
    /// - at least one layer is present;
    /// - each layer has `weights.len() == input_size * output_size` and
    ///   `biases.len() == output_size`;
    /// - `layer[n].output_size == layer[n + 1].input_size` for every pair.
    pub fn validate(&self) -> Result<(), crate::error::ZkmlError> {
        if self.layers.is_empty() {
            return Err(crate::error::ZkmlError::InvalidModel(
                "TinyMLP must have at least one layer".into(),
            ));
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let expected_weights = layer.input_size.saturating_mul(layer.output_size);
            if layer.weights.len() != expected_weights {
                return Err(crate::error::ZkmlError::InvalidModel(format!(
                    "layer {i}: expected {expected_weights} weights, got {}",
                    layer.weights.len()
                )));
            }
            if layer.biases.len() != layer.output_size {
                return Err(crate::error::ZkmlError::InvalidModel(format!(
                    "layer {i}: expected {} biases, got {}",
                    layer.output_size,
                    layer.biases.len()
                )));
            }
        }
        for i in 0..self.layers.len().saturating_sub(1) {
            let out = self.layers[i].output_size;
            let next_in = self.layers[i + 1].input_size;
            if out != next_in {
                return Err(crate::error::ZkmlError::InvalidModel(format!(
                    "layer {i} output_size {out} does not match layer {} input_size {next_in}",
                    i + 1
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Unified enum
// ---------------------------------------------------------------------------

/// Enum over supported model architectures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Model {
    DecisionTree(DecisionTree),
    LogisticRegression(LogisticRegression),
    TinyMLP(TinyMLP),
}

impl DecisionTree {
    /// Validate that every split node references in-bounds children and
    /// feature indices. Returns an error describing the first problem found.
    pub fn validate(&self) -> Result<(), crate::error::ZkmlError> {
        for (i, node) in self.nodes.iter().enumerate() {
            if let TreeNode::Split {
                feature_index,
                left,
                right,
                ..
            } = node
            {
                if *feature_index >= self.num_features {
                    return Err(crate::error::ZkmlError::InvalidModel(format!(
                        "node {i}: feature index {feature_index} out of range"
                    )));
                }
                if *left >= self.nodes.len() || *right >= self.nodes.len() {
                    return Err(crate::error::ZkmlError::InvalidModel(format!(
                        "node {i}: child index out of range"
                    )));
                }
            }
        }
        Ok(())
    }
}

impl LogisticRegression {
    /// Number of input features this model expects.
    pub fn num_features(&self) -> usize {
        self.weights.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed_point::FixedPoint;

    #[test]
    fn valid_tree_passes() {
        let tree = DecisionTree {
            num_features: 1,
            nodes: vec![
                TreeNode::Split {
                    feature_index: 0,
                    threshold: FixedPoint::quantize(0.5),
                    left: 1,
                    right: 2,
                },
                TreeNode::Leaf {
                    value: FixedPoint::quantize(0.0),
                },
                TreeNode::Leaf {
                    value: FixedPoint::quantize(1.0),
                },
            ],
        };
        assert!(tree.validate().is_ok());
    }

    #[test]
    fn out_of_range_child_fails() {
        let tree = DecisionTree {
            num_features: 1,
            nodes: vec![TreeNode::Split {
                feature_index: 0,
                threshold: FixedPoint::quantize(0.5),
                left: 9,
                right: 2,
            }],
        };
        assert!(tree.validate().is_err());
    }
}

impl Model {
    /// Number of input features this model expects.
    pub fn num_features(&self) -> usize {
        match self {
            Model::DecisionTree(t) => t.num_features,
            Model::LogisticRegression(lr) => lr.weights.len(),
            Model::TinyMLP(m) => m.layers.first().map(|l| l.input_size).unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests_features {
    use super::*;
    use crate::fixed_point::FixedPoint;

    #[test]
    fn logistic_feature_count() {
        let model = Model::LogisticRegression(LogisticRegression {
            weights: vec![FixedPoint::quantize(0.1); 5],
            bias: FixedPoint::quantize(0.0),
        });
        assert_eq!(model.num_features(), 5);
    }
}
