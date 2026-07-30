//! Golden-vector test suite for the inference engine.
//!
//! This test suite loads model and input vectors from JSON files and verifies
//! that the inference engine produces the expected outputs. The same vectors
//! can be reused by RISC Zero guest tests and Phase 2 circuit tests.
//!
//! Vectors are stored in the `vectors/` subdirectory as JSON files with the
//! following structure:
//! ```json
//! {
//!   "description": "Human-readable description of the test case",
//!   "model": { ... },
//!   "test_cases": [
//!     {
//!       "description": "Description of this specific test case",
//!       "inputs": [ ... ],
//!       "expected_output": { ... },
//!       "expected_error": "Optional: error type if this should fail"
//!     }
//!   ]
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use zkml_common::fixed_point::FixedPoint;
use zkml_common::models::{DecisionTree, LogisticRegression, Model, TreeNode};
use zkml_prover::inference::{run_inference, try_run_inference};

/// Fixed-point representation matching the Rust struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FixedPointJson {
    value: i64,
    scale: u32,
}

impl From<FixedPointJson> for FixedPoint {
    fn from(fp: FixedPointJson) -> Self {
        FixedPoint::from_raw(fp.value, fp.scale)
    }
}

/// Decision tree node representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum TreeNodeJson {
    Split {
        feature_index: usize,
        threshold: FixedPointJson,
        left: usize,
        right: usize,
    },
    Leaf {
        value: FixedPointJson,
    },
}

impl From<TreeNodeJson> for TreeNode {
    fn from(node: TreeNodeJson) -> Self {
        match node {
            TreeNodeJson::Split {
                feature_index,
                threshold,
                left,
                right,
            } => TreeNode::Split {
                feature_index,
                threshold: threshold.into(),
                left,
                right,
            },
            TreeNodeJson::Leaf { value } => TreeNode::Leaf {
                value: value.into(),
            },
        }
    }
}

/// Decision tree model representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecisionTreeJson {
    #[serde(rename = "type")]
    model_type: String,
    num_features: usize,
    nodes: Vec<TreeNodeJson>,
}

/// Logistic regression model representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LogisticRegressionJson {
    #[serde(rename = "type")]
    model_type: String,
    weights: Vec<FixedPointJson>,
    bias: FixedPointJson,
}

/// Dense layer representation for TinyMLP.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DenseLayerJson {
    weights: Vec<FixedPointJson>,
    biases: Vec<FixedPointJson>,
    input_size: usize,
    output_size: usize,
}

/// TinyMLP model representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TinyMLPJson {
    #[serde(rename = "type")]
    model_type: String,
    layers: Vec<DenseLayerJson>,
}

/// Model representation (enum over all supported types).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum ModelJson {
    DecisionTree(DecisionTreeJson),
    LogisticRegression(LogisticRegressionJson),
    TinyMLP(TinyMLPJson),
}

impl TryFrom<ModelJson> for Model {
    type Error = String;

    fn try_from(model: ModelJson) -> Result<Self, String> {
        match model {
            ModelJson::DecisionTree(tree) => Ok(Model::DecisionTree(DecisionTree {
                nodes: tree.nodes.into_iter().map(|n| n.into()).collect(),
                num_features: tree.num_features,
            })),
            ModelJson::LogisticRegression(lr) => {
                Ok(Model::LogisticRegression(LogisticRegression {
                    weights: lr.weights.into_iter().map(|w| w.into()).collect(),
                    bias: lr.bias.into(),
                }))
            }
            ModelJson::TinyMLP(mlp) => {
                use zkml_common::models::DenseLayer;
                let layers: Result<Vec<DenseLayer>, String> = mlp
                    .layers
                    .into_iter()
                    .map(|l| {
                        Ok(DenseLayer {
                            weights: l.weights.into_iter().map(|w| w.into()).collect(),
                            biases: l.biases.into_iter().map(|b| b.into()).collect(),
                            input_size: l.input_size,
                            output_size: l.output_size,
                        })
                    })
                    .collect();
                Ok(Model::TinyMLP(zkml_common::models::TinyMLP {
                    layers: layers?,
                }))
            }
        }
    }
}

/// Test case representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestCaseJson {
    description: String,
    inputs: Vec<FixedPointJson>,
    #[serde(default)]
    expected_output: Option<FixedPointJson>,
    #[serde(default)]
    expected_error: Option<String>,
}

/// Complete test vector file.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestVectorJson {
    description: String,
    model: ModelJson,
    test_cases: Vec<TestCaseJson>,
}

/// Get the path to a test vector file relative to the tests directory.
fn vector_path(filename: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("vectors");
    path.push(filename);
    path
}

/// Load and parse a test vector JSON file.
fn load_test_vector(path: &Path) -> TestVectorJson {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Failed to parse JSON from {}: {}", path.display(), e))
}

/// Run a single test case and verify the result.
fn run_test_case(model: &Model, test_case: &TestCaseJson) {
    let inputs: Vec<FixedPoint> = test_case
        .inputs
        .iter()
        .map(|fp| fp.clone().into())
        .collect();

    if let Some(expected_error) = &test_case.expected_error {
        // This test case should error - route through try_run_inference for typed errors
        // For InvalidModel errors, validate the model first
        if expected_error.contains("InvalidModel") {
            if let Model::DecisionTree(tree) = model {
                let result = tree.validate();
                match result {
                    Ok(_) => panic!(
                        "Expected InvalidModel error but validation succeeded for: {}",
                        test_case.description
                    ),
                    Err(e) => {
                        let error_str = format!("{:?}", e);
                        if !error_str.contains(expected_error) {
                            panic!(
                                "Expected error containing '{}' but got: {}",
                                expected_error, error_str
                            );
                        }
                    }
                }
                return;
            }
        }

        let result = try_run_inference(model, &inputs);

        match result {
            Ok(_) => panic!(
                "Expected error '{}' but inference succeeded for: {}",
                expected_error, test_case.description
            ),
            Err(e) => {
                let error_str = format!("{:?}", e);
                if !error_str.contains(expected_error) {
                    panic!(
                        "Expected error containing '{}' but got: {}",
                        expected_error, error_str
                    );
                }
            }
        }
    } else {
        // This test case should succeed
        let output = run_inference(model, &inputs);
        let expected = test_case.expected_output.as_ref().unwrap();
        let expected_fp: FixedPoint = expected.clone().into();

        assert_eq!(
            output.value, expected_fp.value,
            "Output mismatch for: {} (got {}, expected {})",
            test_case.description, output.value, expected_fp.value
        );
        assert_eq!(
            output.scale, expected_fp.scale,
            "Scale mismatch for: {}",
            test_case.description
        );
    }
}

/// Macro to generate a test for each vector file.
macro_rules! generate_vector_tests {
    ($($file:ident),*) => {
        $(
            #[test]
            fn $file() {
                let path = vector_path(concat!(stringify!($file), ".json"));
                let vector = load_test_vector(&path);

                let model: Model = vector.model.try_into()
                    .unwrap_or_else(|e| panic!("Failed to convert model: {}", e));

                // Validate decision trees - this catches InvalidModel errors before inference
                // For InvalidModel tests, we validate during the test case itself
                let has_invalid_model_error = vector.test_cases.iter().any(|tc| {
                    tc.expected_error.as_ref().map(|e| e.contains("InvalidModel")).unwrap_or(false)
                });
                if !has_invalid_model_error {
                    if let Model::DecisionTree(tree) = &model {
                        tree.validate().unwrap_or_else(|e| {
                            panic!("Tree validation failed for {}: {:?}", path.display(), e)
                        });
                    }
                }

                for test_case in &vector.test_cases {
                    run_test_case(&model, test_case);
                }
            }
        )*
    };
}

// Generate tests for all vector files
generate_vector_tests!(
    decision_tree_depth1,
    decision_tree_depth3,
    decision_tree_boundary,
    decision_tree_degenerate,
    logistic_regression_positive,
    logistic_regression_negative,
    logistic_regression_zero,
    logistic_regression_rounding,
    error_feature_mismatch,
    error_out_of_range_child
);
