//! One-shot helper that writes the ONNX fixture files used by import tests.
//!
//! ```text
//! cargo run -p zkml-prover --example generate_onnx_fixtures
//! ```
//!
//! Fixtures are synthetic `ModelProto` encodings (same field tags as real ONNX
//! files). Opset pairs mirror real exporters: core `ai.onnx` >= 17 and
//! `ai.onnx.ml` in 1–5 (never ml=17).

use std::fs;
use std::path::PathBuf;

use prost::Message;
use zkml_prover::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto};

fn encode(model: &ModelProto) -> Vec<u8> {
    let mut buf = Vec::new();
    model.encode(&mut buf).expect("encode");
    buf
}

fn node(name: &str, op_type: &str, domain: &str) -> NodeProto {
    NodeProto {
        name: name.into(),
        op_type: op_type.into(),
        domain: domain.into(),
        input: vec!["X".into()],
        output: vec!["Y".into()],
        attribute: vec![],
    }
}

fn linear_classifier_node(name: &str, coefficients: Vec<f32>, intercepts: Vec<f32>) -> NodeProto {
    NodeProto {
        name: name.into(),
        op_type: "LinearClassifier".into(),
        domain: "ai.onnx.ml".into(),
        input: vec!["X".into()],
        output: vec!["Y".into()],
        attribute: vec![
            AttributeProto {
                name: "coefficients".into(),
                floats: coefficients,
                f: 0.0,
                i: 0,
                ints: vec![],
                s: vec![],
                strings: vec![],
                t: None,
                g: None,
                sparse_tensor: None,
                r#type: 0,
            },
            AttributeProto {
                name: "intercepts".into(),
                floats: intercepts,
                f: 0.0,
                i: 0,
                ints: vec![],
                s: vec![],
                strings: vec![],
                t: None,
                g: None,
                sparse_tensor: None,
                r#type: 0,
            },
            AttributeProto {
                name: "post_transform".into(),
                floats: vec![],
                f: 0.0,
                i: 0,
                ints: vec![],
                s: b"NONE".to_vec(),
                strings: vec![],
                t: None,
                g: None,
                sparse_tensor: None,
                r#type: 0,
            },
        ],
    }
}

fn opsets(core: i64, ml: i64) -> Vec<OperatorSetIdProto> {
    vec![
        OperatorSetIdProto {
            domain: String::new(),
            version: core,
        },
        OperatorSetIdProto {
            domain: "ai.onnx.ml".into(),
            version: ml,
        },
    ]
}

fn main() {
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    fs::create_dir_all(&out).expect("create fixtures dir");

    // Valid decision-tree: realistic skl2onnx-like pair (core 17 + ml 3).
    let tree = ModelProto {
        ir_version: 8,
        producer_name: "zkml-fixture-generator".into(),
        producer_version: "0.1".into(),
        opset_import: opsets(17, 3),
        graph: Some(GraphProto {
            name: "decision_tree".into(),
            node: vec![node(
                "tree_ensemble",
                "TreeEnsembleClassifier",
                "ai.onnx.ml",
            )],
        }),
        ..Default::default()
    };
    fs::write(out.join("decision_tree_valid.onnx"), encode(&tree)).unwrap();

    // Explicit skl2onnx-like layout (same graph; distinct name for regression).
    let skl = ModelProto {
        ir_version: 8,
        producer_name: "skl2onnx".into(),
        producer_version: "1.16".into(),
        opset_import: opsets(17, 3),
        graph: Some(GraphProto {
            name: "SklearnDecisionTreeClassifier".into(),
            node: vec![node(
                "TreeEnsembleClassifier",
                "TreeEnsembleClassifier",
                "ai.onnx.ml",
            )],
        }),
        ..Default::default()
    };
    fs::write(out.join("skl2onnx_like_tree.onnx"), encode(&skl)).unwrap();

    // Intentionally unsupported operator (Conv) with a valid core opset.
    let unsupported = ModelProto {
        ir_version: 8,
        producer_name: "zkml-fixture-generator".into(),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        graph: Some(GraphProto {
            name: "cnn".into(),
            node: vec![node("conv0", "Conv", "")],
        }),
        ..Default::default()
    };
    fs::write(out.join("unsupported_conv.onnx"), encode(&unsupported)).unwrap();

    // Core domain below floor (ml is fine at 3).
    let low_opset = ModelProto {
        ir_version: 7,
        producer_name: "zkml-fixture-generator".into(),
        opset_import: opsets(13, 3),
        graph: Some(GraphProto {
            name: "old_tree".into(),
            node: vec![node(
                "tree_ensemble",
                "TreeEnsembleClassifier",
                "ai.onnx.ml",
            )],
        }),
        ..Default::default()
    };
    fs::write(out.join("low_opset_tree.onnx"), encode(&low_opset)).unwrap();

    // Linear classifier: core 18 + ml 1 (LinearClassifier is ml opset 1 only).
    // Binary classifier with 3 features.
    let linear = ModelProto {
        ir_version: 8,
        producer_name: "zkml-fixture-generator".into(),
        opset_import: opsets(18, 1),
        graph: Some(GraphProto {
            name: "logistic".into(),
            node: vec![linear_classifier_node(
                "linear",
                vec![0.5, -0.3, 0.8], // 3 coefficients
                vec![0.1],            // 1 intercept (binary)
            )],
        }),
        ..Default::default()
    };
    fs::write(out.join("linear_classifier_valid.onnx"), encode(&linear)).unwrap();

    println!("wrote fixtures to {}", out.display());
}
