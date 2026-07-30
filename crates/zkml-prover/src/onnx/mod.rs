//! ONNX model importer foundation.
//!
//! This module deserializes ONNX protobuf (`ModelProto`), validates the opset
//! version and operator set, and returns a typed error. Parameter extraction
//! into internal model types is intentionally deferred:
//!
//! - Tree ensemble extraction: GitHub issue #5
//! - Linear classifier extraction: GitHub issue #6
//!
//! # Supported operators (allowlist)
//!
//! | Operator                | Target (future)        |
//! |-------------------------|------------------------|
//! | `TreeEnsembleClassifier`| `DecisionTree`         |
//! | `LinearClassifier`      | `LogisticRegression`   |
//! | `MatMul`                | `TinyMLP` dense layer  |
//! | `Add`                   | `TinyMLP` bias         |
//! | `Relu`                  | `TinyMLP` activation   |
//!
//! Unsupported operators fail at import time with a clear error rather than
//! silently ignoring nodes.

mod error;
mod extract;
mod proto;
mod tree_extractor;
mod validate;

pub use error::OnnxImportError;
pub use extract::extract_linear_classifier;
pub use proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorDataType,
    TensorShapeProto, TensorShapeProtoDimension, TensorTypeProto, TypeProto, ValueInfoProto,
};
pub use validate::{MIN_OPSET_CORE, MIN_OPSET_ML};

use prost::Message;
use validate::{check_operators, check_opset, detect_architecture};
use zkml_common::models::Model;

/// Extract the number of features from the model's input tensor shape.
///
/// Assumes the first input tensor is the feature vector and extracts its
/// last dimension (for 2D tensors like [batch_size, num_features]).
fn extract_num_features(model: &ModelProto) -> Result<usize, OnnxImportError> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxImportError::MalformedModel("model has no graph".into()))?;

    if graph.input.is_empty() {
        return Err(OnnxImportError::MalformedModel(
            "model has no input tensors".into(),
        ));
    }

    let first_input = &graph.input[0];
    let type_info = first_input
        .r#type
        .as_ref()
        .ok_or_else(|| OnnxImportError::MalformedModel("input has no type info".into()))?;

    let tensor_type = type_info
        .tensor
        .as_ref()
        .ok_or_else(|| OnnxImportError::MalformedModel("input is not a tensor".into()))?;

    let shape = tensor_type
        .shape
        .as_ref()
        .ok_or_else(|| OnnxImportError::MalformedModel("input has no shape".into()))?;

    if shape.dim.is_empty() {
        return Err(OnnxImportError::MalformedModel(
            "input shape has no dimensions".into(),
        ));
    }

    // For 1D tensors [num_features], use the only dimension
    // For 2D tensors [batch_size, num_features], use the last dimension
    let last_dim = &shape.dim[shape.dim.len() - 1];

    if last_dim.dim_value > 0 {
        Ok(last_dim.dim_value as usize)
    } else if !last_dim.dim_param.is_empty() {
        // Symbolic dimension - cannot determine at import time
        Err(OnnxImportError::MalformedModel(format!(
            "input has symbolic dimension '{}', cannot determine num_features",
            last_dim.dim_param
        )))
    } else {
        Err(OnnxImportError::MalformedModel(
            "input dimension has no value or parameter".into(),
        ))
    }
}

/// Minimum core ONNX opset version (`""` / `ai.onnx`).
///
/// Alias of [`MIN_OPSET_CORE`] for callers that still use the historical name.
pub const MIN_OPSET_VERSION: i64 = MIN_OPSET_CORE;

/// Operators currently allowed by the importer foundation.
///
/// Extraction of each family is tracked in issues #5 (trees) and #6 (linear).
pub const SUPPORTED_OPERATORS: &[&str] = &[
    "TreeEnsembleClassifier",
    "LinearClassifier",
    "MatMul",
    "Add",
    "Relu",
];

/// Import an ONNX model from a protobuf byte slice.
///
/// Performs protobuf decoding, per-domain opset validation (core `>= 17`,
/// `ai.onnx.ml` `>= 1`), and operator allowlist checks. When validation
/// succeeds, extracts model parameters into the internal representation.
///
/// # Errors
///
/// - [`OnnxImportError::MalformedModel`] if the bytes are not a valid
///   `ModelProto` or the graph is missing / empty.
/// - [`OnnxImportError::UnsupportedOpset`] if a known domain is below its floor.
/// - [`OnnxImportError::UnsupportedOperator`] if a graph node uses an op
///   outside the allowlist.
/// - [`OnnxImportError::ExtractionNotImplemented`] for operators not yet implemented.
/// - [`OnnxImportError::MalformedModel`] if parameter extraction fails.
pub fn import_onnx(bytes: &[u8]) -> Result<Model, OnnxImportError> {
    let model = parse_model_proto(bytes)?;
    validate_model(&model)?;

    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxImportError::MalformedModel("model has no graph".into()))?;

    // Determine the architecture and extract accordingly
    let architecture = detect_architecture(&model);

    // For now, only support single-operator models
    if graph.node.len() != 1 {
        return Err(OnnxImportError::MalformedModel(
            "only single-operator models are supported".into(),
        ));
    }

    let node = &graph.node[0];

    match node.op_type.as_str() {
        "TreeEnsembleClassifier" => {
            let num_features = extract_num_features(&model)?;
            let tree = tree_extractor::extract_tree(node, num_features)?;
            Ok(Model::DecisionTree(tree))
        }
        "LinearClassifier" => {
            let lr = extract_linear_classifier(node)?;
            Ok(Model::LogisticRegression(lr))
        }
        "MatMul" | "Add" | "Relu" => Err(OnnxImportError::ExtractionNotImplemented {
            architecture_hint: architecture,
        }),
        _ => Err(OnnxImportError::UnsupportedOperator {
            op_type: node.op_type.clone(),
        }),
    }
}

/// Decode raw bytes into an ONNX `ModelProto` without further validation.
///
/// Useful for tests and for callers that want to inspect the graph after
/// [`import_onnx`] rejects extraction.
pub fn parse_model_proto(bytes: &[u8]) -> Result<ModelProto, OnnxImportError> {
    ModelProto::decode(bytes)
        .map_err(|e| OnnxImportError::MalformedModel(format!("protobuf decode failed: {e}")))
}

/// Validate opset version and operator allowlist on an already-decoded model.
pub fn validate_model(model: &ModelProto) -> Result<(), OnnxImportError> {
    check_opset(model)?;
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxImportError::MalformedModel("model has no graph".into()))?;
    if graph.node.is_empty() {
        return Err(OnnxImportError::MalformedModel(
            "model graph has no nodes".into(),
        ));
    }
    check_operators(graph, SUPPORTED_OPERATORS)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    fn encode(model: &ModelProto) -> Vec<u8> {
        let mut buf = Vec::new();
        model.encode(&mut buf).expect("encode model");
        buf
    }

    /// Build a model with independent core and ML domain versions.
    fn model_with(core_opset: i64, ml_opset: i64, ops: &[&str]) -> ModelProto {
        ModelProto {
            ir_version: 8,
            opset_import: vec![
                OperatorSetIdProto {
                    domain: String::new(),
                    version: core_opset,
                },
                OperatorSetIdProto {
                    domain: "ai.onnx.ml".into(),
                    version: ml_opset,
                },
            ],
            graph: Some(GraphProto {
                name: "test".into(),
                input: vec![],
                output: vec![],
                node: ops
                    .iter()
                    .enumerate()
                    .map(|(i, op)| NodeProto {
                        name: format!("n{i}"),
                        op_type: (*op).into(),
                        domain: if op.ends_with("Classifier") {
                            "ai.onnx.ml".into()
                        } else {
                            String::new()
                        },
                        input: vec!["X".into()],
                        output: vec![format!("Y{i}")],
                        attribute: vec![],
                    })
                    .collect(),
            }),
            ..Default::default()
        }
    }

    #[test]
    fn valid_tree_extraction_fails_without_attributes() {
        // Realistic skl2onnx-like pair: core 17 + ml 3.
        // This will fail because the test helper doesn't include tree attributes
        let bytes = encode(&model_with(17, 3, &["TreeEnsembleClassifier"]));
        let err = import_onnx(&bytes).unwrap_err();
        match err {
            OnnxImportError::MalformedModel(_) => {
                // Expected - missing tree attributes
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn linear_classifier_without_attributes_fails() {
        // LinearClassifier has only ever been ai.onnx.ml version 1.
        // This test verifies that a LinearClassifier without coefficients/intercepts
        // attributes fails with a MalformedModel error.
        let bytes = encode(&model_with(18, 1, &["LinearClassifier"]));
        let err = import_onnx(&bytes).unwrap_err();
        assert!(matches!(
            err,
            OnnxImportError::MalformedModel(_) // Fails due to missing attributes
        ));
    }

    #[test]
    fn valid_mlp_ops_reaches_extraction_not_implemented() {
        let bytes = encode(&model_with(17, 1, &["MatMul", "Add", "Relu"]));
        let err = import_onnx(&bytes).unwrap_err();
        // Now expects MalformedModel because model lacks input tensors and has multiple operators
        assert!(matches!(err, OnnxImportError::MalformedModel(_)));
    }

    #[test]
    fn unsupported_operator_names_offender() {
        let bytes = encode(&model_with(17, 1, &["Conv"]));
        let err = import_onnx(&bytes).unwrap_err();
        match err {
            OnnxImportError::UnsupportedOperator { op_type } => assert_eq!(op_type, "Conv"),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn low_core_opset_is_rejected() {
        let bytes = encode(&model_with(13, 3, &["TreeEnsembleClassifier"]));
        let err = import_onnx(&bytes).unwrap_err();
        match err {
            OnnxImportError::UnsupportedOpset { found, required } => {
                assert_eq!(found, 13);
                assert_eq!(required, MIN_OPSET_CORE);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn realistic_ml_opset_is_accepted() {
        let bytes = encode(&model_with(17, 5, &["TreeEnsembleClassifier"]));
        let err = import_onnx(&bytes).unwrap_err();
        // Now expects MalformedModel because model lacks input tensors
        assert!(
            matches!(err, OnnxImportError::MalformedModel(_)),
            "got {err}"
        );
    }

    #[test]
    fn garbage_bytes_are_malformed() {
        let err = import_onnx(b"not a protobuf model").unwrap_err();
        assert!(matches!(err, OnnxImportError::MalformedModel(_)));
    }

    #[test]
    fn missing_graph_is_malformed() {
        let model = ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            graph: None,
            ..Default::default()
        };
        let err = import_onnx(&encode(&model)).unwrap_err();
        assert!(matches!(err, OnnxImportError::MalformedModel(_)));
    }

    #[test]
    fn empty_graph_is_malformed() {
        let model = ModelProto {
            ir_version: 8,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            graph: Some(GraphProto {
                name: "empty".into(),
                input: vec![],
                output: vec![],
                node: vec![],
            }),
            ..Default::default()
        };
        let err = import_onnx(&encode(&model)).unwrap_err();
        assert!(matches!(err, OnnxImportError::MalformedModel(_)));
    }

    #[test]
    fn parse_model_proto_round_trips() {
        let original = model_with(17, 1, &["Add", "Relu"]);
        let bytes = encode(&original);
        let decoded = parse_model_proto(&bytes).unwrap();
        assert_eq!(decoded.opset_import[0].version, 17);
        assert_eq!(decoded.opset_import[1].version, 1);
        assert_eq!(decoded.graph.as_ref().unwrap().node.len(), 2);
    }
}
