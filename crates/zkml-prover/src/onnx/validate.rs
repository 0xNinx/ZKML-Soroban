//! Opset and operator validation for ONNX models.

use super::error::OnnxImportError;
use super::proto::{GraphProto, ModelProto};

/// Minimum opset for the core ONNX domain (`""` / `ai.onnx`).
pub const MIN_OPSET_CORE: i64 = 17;

/// Minimum opset for the classic ML domain (`ai.onnx.ml`).
///
/// This domain versions independently of the core domain (current range is
/// roughly 1–5). There is no `ai.onnx.ml` opset 17.
pub const MIN_OPSET_ML: i64 = 1;

/// Per-domain opset floor.
///
/// Returns `None` for domains we do not enforce (e.g. vendor extensions).
pub fn min_version_for_domain(domain: &str) -> Option<i64> {
    match domain {
        "" | "ai.onnx" => Some(MIN_OPSET_CORE),
        "ai.onnx.ml" => Some(MIN_OPSET_ML),
        _ => None,
    }
}

/// Ensure every known domain's opset import meets its floor.
///
/// If the model declares no opset imports at all, treat that as malformed:
/// well-formed ONNX models always list their opset requirements.
pub fn check_opset(model: &ModelProto) -> Result<(), OnnxImportError> {
    if model.opset_import.is_empty() {
        return Err(OnnxImportError::MalformedModel(
            "model declares no opset_import entries".into(),
        ));
    }

    let mut saw_relevant = false;
    for entry in &model.opset_import {
        let Some(required) = min_version_for_domain(&entry.domain) else {
            continue;
        };
        saw_relevant = true;
        if entry.version < required {
            return Err(OnnxImportError::UnsupportedOpset {
                found: entry.version,
                required,
            });
        }
    }

    // If only exotic domains were listed, still require an explicit core/ml
    // opset so we never silently accept an unspecified dialect.
    if !saw_relevant {
        return Err(OnnxImportError::MalformedModel(
            "model has no opset_import for ai.onnx or ai.onnx.ml".into(),
        ));
    }

    Ok(())
}

/// Walk every graph node and reject the first operator outside `allowed`.
pub fn check_operators(graph: &GraphProto, allowed: &[&str]) -> Result<(), OnnxImportError> {
    for node in &graph.node {
        let op = node.op_type.as_str();
        if op.is_empty() {
            return Err(OnnxImportError::MalformedModel(format!(
                "node '{}' has empty op_type",
                node.name
            )));
        }
        if !allowed.contains(&op) {
            return Err(OnnxImportError::UnsupportedOperator {
                op_type: op.to_string(),
            });
        }
    }
    Ok(())
}

/// Derive a short architecture label from the operator mix for error messages.
pub fn detect_architecture(model: &ModelProto) -> String {
    let Some(graph) = model.graph.as_ref() else {
        return "unknown".into();
    };
    let mut has_tree = false;
    let mut has_linear = false;
    let mut has_matmul = false;
    let mut has_add = false;
    let mut has_relu = false;

    for node in &graph.node {
        match node.op_type.as_str() {
            "TreeEnsembleClassifier" => has_tree = true,
            "LinearClassifier" => has_linear = true,
            "MatMul" => has_matmul = true,
            "Add" => has_add = true,
            "Relu" => has_relu = true,
            _ => {}
        }
    }

    if has_tree {
        "decision tree (TreeEnsembleClassifier)".into()
    } else if has_linear {
        "logistic regression (LinearClassifier)".into()
    } else if has_matmul {
        let mut parts = vec!["MatMul"];
        if has_add {
            parts.push("Add");
        }
        if has_relu {
            parts.push("Relu");
        }
        format!("tiny MLP ({})", parts.join(" + "))
    } else {
        "unknown".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::proto::{NodeProto, OperatorSetIdProto};

    #[test]
    fn rejects_empty_opset_list() {
        let model = ModelProto::default();
        assert!(matches!(
            check_opset(&model),
            Err(OnnxImportError::MalformedModel(_))
        ));
    }

    #[test]
    fn ignores_unrelated_domains_when_core_present() {
        let model = ModelProto {
            opset_import: vec![
                OperatorSetIdProto {
                    domain: "com.example".into(),
                    version: 1,
                },
                OperatorSetIdProto {
                    domain: String::new(),
                    version: 17,
                },
            ],
            ..Default::default()
        };
        assert!(check_opset(&model).is_ok());
    }

    #[test]
    fn accepts_realistic_ml_opset_with_core_17() {
        let model = ModelProto {
            opset_import: vec![
                OperatorSetIdProto {
                    domain: String::new(),
                    version: 17,
                },
                OperatorSetIdProto {
                    domain: "ai.onnx.ml".into(),
                    version: 3,
                },
            ],
            ..Default::default()
        };
        assert!(check_opset(&model).is_ok());
    }

    #[test]
    fn rejects_core_below_floor() {
        let model = ModelProto {
            opset_import: vec![
                OperatorSetIdProto {
                    domain: String::new(),
                    version: 13,
                },
                OperatorSetIdProto {
                    domain: "ai.onnx.ml".into(),
                    version: 3,
                },
            ],
            ..Default::default()
        };
        match check_opset(&model).unwrap_err() {
            OnnxImportError::UnsupportedOpset { found, required } => {
                assert_eq!(found, 13);
                assert_eq!(required, MIN_OPSET_CORE);
            }
            other => panic!("unexpected: {other}"),
        }
    }

    #[test]
    fn rejects_ml_below_floor() {
        let model = ModelProto {
            opset_import: vec![
                OperatorSetIdProto {
                    domain: String::new(),
                    version: 17,
                },
                OperatorSetIdProto {
                    domain: "ai.onnx.ml".into(),
                    version: 0,
                },
            ],
            ..Default::default()
        };
        match check_opset(&model).unwrap_err() {
            OnnxImportError::UnsupportedOpset { found, required } => {
                assert_eq!(found, 0);
                assert_eq!(required, MIN_OPSET_ML);
            }
            other => panic!("unexpected: {other}"),
        }
    }

    #[test]
    fn operator_check_reports_first_offender() {
        let graph = GraphProto {
            name: "g".into(),
            node: vec![
                NodeProto {
                    op_type: "MatMul".into(),
                    attribute: vec![],
                    ..Default::default()
                },
                NodeProto {
                    op_type: "Softmax".into(),
                    attribute: vec![],
                    ..Default::default()
                },
            ],
        };
        match check_operators(&graph, &["MatMul", "Add", "Relu"]).unwrap_err() {
            OnnxImportError::UnsupportedOperator { op_type } => {
                assert_eq!(op_type, "Softmax");
            }
            other => panic!("unexpected: {other}"),
        }
    }
}
