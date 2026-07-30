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
mod validate;

pub use error::OnnxImportError;
pub use extract::extract_linear_classifier;
pub use proto::{AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto};
pub use validate::{MIN_OPSET_CORE, MIN_OPSET_ML};

use prost::Message;
use validate::{check_operators, check_opset, detect_architecture};
use zkml_common::models::Model;

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
/// - [`OnnxImportError::MalformedModel`] if parameter extraction fails.
pub fn import_onnx(bytes: &[u8]) -> Result<Model, OnnxImportError> {
    let model = parse_model_proto(bytes)?;
    validate_model(&model)?;

    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxImportError::MalformedModel("model has no graph".into()))?;

    // For now, we only support single-operator graphs
    if graph.node.len() != 1 {
        return Err(OnnxImportError::MalformedModel(format!(
            "Expected exactly 1 node, found {}",
            graph.node.len()
        )));
    }

    let node = &graph.node[0];

    match node.op_type.as_str() {
        "LinearClassifier" => {
            let lr = extract_linear_classifier(node)?;
            Ok(Model::LogisticRegression(lr))
        }
        "TreeEnsembleClassifier" => {
            let hint = detect_architecture(&model);
            Err(OnnxImportError::ExtractionNotImplemented {
                architecture_hint: hint,
            })
        }
        _ => {
            let hint = detect_architecture(&model);
            Err(OnnxImportError::ExtractionNotImplemented {
                architecture_hint: hint,
            })
        }
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
    fn valid_tree_reaches_extraction_not_implemented() {
        // Realistic skl2onnx-like pair: core 17 + ml 3.
        let bytes = encode(&model_with(17, 3, &["TreeEnsembleClassifier"]));
        let err = import_onnx(&bytes).unwrap_err();
        match err {
            OnnxImportError::ExtractionNotImplemented { architecture_hint } => {
                assert!(
                    architecture_hint.to_lowercase().contains("tree"),
                    "hint was: {architecture_hint}"
                );
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
        // MLP has multiple nodes, which is not supported yet (single-operator graphs only)
        assert!(matches!(
            err,
            OnnxImportError::MalformedModel(_) // Expected exactly 1 node, found 3
        ));
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
        assert!(
            matches!(err, OnnxImportError::ExtractionNotImplemented { .. }),
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
