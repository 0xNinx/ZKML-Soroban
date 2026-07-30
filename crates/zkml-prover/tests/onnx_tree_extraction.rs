//! End-to-end tests for TreeEnsembleClassifier extraction.
//!
//! Tests the complete flow from ONNX bytes to DecisionTree model,
//! including structural validation and inference correctness.

use zkml_common::fixed_point::FixedPoint;
use zkml_common::inference::run_inference;
use zkml_common::models::Model;
use zkml_prover::onnx;

/// Create a minimal valid TreeEnsembleClassifier ONNX model for testing.
fn create_test_tree_onnx() -> Vec<u8> {
    use prost::Message;
    use zkml_prover::onnx::{
        AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorDataType,
        TensorShapeProto, TensorShapeProtoDimension, TensorTypeProto, TypeProto, ValueInfoProto,
    };

    // Create a simple tree: root splits on feature 0 at threshold 0.5
    // Left child (node 1) is leaf with value 0.0
    // Right child (node 2) is leaf with value 1.0
    ModelProto {
        ir_version: 8,
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
        graph: Some(GraphProto {
            name: "test_tree".into(),
            input: vec![ValueInfoProto {
                name: "X".into(),
                r#type: Some(TypeProto {
                    tensor: Some(TensorTypeProto {
                        elem_type: TensorDataType::Float as i32,
                        shape: Some(TensorShapeProto {
                            dim: vec![TensorShapeProtoDimension {
                                dim_value: 1, // num_features
                                dim_param: String::new(),
                            }],
                        }),
                    }),
                }),
            }],
            output: vec![],
            node: vec![NodeProto {
                name: "tree".into(),
                op_type: "TreeEnsembleClassifier".into(),
                domain: "ai.onnx.ml".into(),
                input: vec!["X".into()],
                output: vec!["Y".into()],
                attribute: vec![
                    // Tree structure attributes
                    AttributeProto {
                        name: "nodes_treeids".into(),
                        ints: vec![0, 0, 0], // All nodes in tree 0
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_nodeids".into(),
                        ints: vec![0, 1, 2], // Node IDs
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_featureids".into(),
                        ints: vec![0, 0, 0], // 3 nodes
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_values".into(),
                        floats: vec![0.5, 0.0, 0.0], // Thresholds (leaf values are 0 in ONNX)
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_modes".into(),
                        strings: vec!["BRANCH_LEQ".into(), "LEAF".into(), "LEAF".into()],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_truenodeids".into(),
                        ints: vec![1, 0, 0], // Child node IDs (not indices)
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_falsenodeids".into(),
                        ints: vec![2, 0, 0], // Child node IDs (not indices)
                        ..Default::default()
                    },
                    // Class attributes for leaf values
                    AttributeProto {
                        name: "class_ids".into(),
                        ints: vec![0, 1], // Class indices
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_weights".into(),
                        floats: vec![0.0, 1.0], // Actual leaf values
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_nodeids".into(),
                        ints: vec![1, 2], // Which leaf nodes these weights belong to
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_treeids".into(),
                        ints: vec![0, 0], // All in tree 0
                        ..Default::default()
                    },
                ],
            }],
        }),
        ..Default::default()
    }
    .encode_to_vec()
}

#[test]
fn test_tree_extraction_round_trip() {
    let onnx_bytes = create_test_tree_onnx();
    let model = onnx::import_onnx(&onnx_bytes).expect("ONNX import should succeed");

    match model {
        Model::DecisionTree(tree) => {
            assert_eq!(tree.nodes.len(), 3);
            assert_eq!(tree.num_features, 1);

            // Verify root node
            match &tree.nodes[0] {
                zkml_common::models::TreeNode::Split {
                    feature_index,
                    threshold,
                    left,
                    right,
                } => {
                    assert_eq!(*feature_index, 0);
                    assert!((threshold.dequantize() - 0.5).abs() < 1e-4);
                    assert_eq!(*left, 1);
                    assert_eq!(*right, 2);
                }
                _ => panic!("Root should be a split node"),
            }

            // Verify leaf nodes
            match &tree.nodes[1] {
                zkml_common::models::TreeNode::Leaf { value } => {
                    assert!((value.dequantize() - 0.0).abs() < 1e-4);
                }
                _ => panic!("Node 1 should be a leaf"),
            }

            match &tree.nodes[2] {
                zkml_common::models::TreeNode::Leaf { value } => {
                    assert!((value.dequantize() - 1.0).abs() < 1e-4);
                }
                _ => panic!("Node 2 should be a leaf"),
            }
        }
        _ => panic!("Expected DecisionTree model"),
    }
}

#[test]
fn test_tree_inference_matches_expected() {
    let onnx_bytes = create_test_tree_onnx();
    let model = onnx::import_onnx(&onnx_bytes).expect("ONNX import should succeed");

    // Test inference with feature value <= 0.5 (should go left to leaf 0.0)
    let inputs = vec![FixedPoint::quantize(0.3)];
    let output = run_inference(&model, &inputs);
    assert!(
        (output.dequantize() - 0.0).abs() < 1e-4,
        "Feature 0.3 should output 0.0"
    );

    // Test inference with feature value > 0.5 (should go right to leaf 1.0)
    let inputs = vec![FixedPoint::quantize(0.7)];
    let output = run_inference(&model, &inputs);
    assert!(
        (output.dequantize() - 1.0).abs() < 1e-4,
        "Feature 0.7 should output 1.0"
    );

    // Test boundary case: feature value == 0.5 (should go left per BRANCH_LEQ)
    let inputs = vec![FixedPoint::quantize(0.5)];
    let output = run_inference(&model, &inputs);
    assert!(
        (output.dequantize() - 0.0).abs() < 1e-4,
        "Feature 0.5 should output 0.0 (LEQ)"
    );
}

#[test]
fn test_reject_multi_tree_ensemble() {
    use prost::Message;
    use zkml_prover::onnx::{
        AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorShapeProto,
        TensorShapeProtoDimension, TensorTypeProto, TypeProto, ValueInfoProto,
    };

    let model = ModelProto {
        ir_version: 8,
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
        graph: Some(GraphProto {
            name: "ensemble".into(),
            input: vec![ValueInfoProto {
                name: "X".into(),
                r#type: Some(TypeProto {
                    tensor: Some(TensorTypeProto {
                        elem_type: 1, // Float
                        shape: Some(TensorShapeProto {
                            dim: vec![TensorShapeProtoDimension {
                                dim_value: 1,
                                dim_param: String::new(),
                            }],
                        }),
                    }),
                }),
            }],
            output: vec![],
            node: vec![NodeProto {
                name: "ensemble".into(),
                op_type: "TreeEnsembleClassifier".into(),
                domain: "ai.onnx.ml".into(),
                input: vec!["X".into()],
                output: vec!["Y".into()],
                attribute: vec![
                    // Two trees indicated by nodes_treeids
                    AttributeProto {
                        name: "nodes_treeids".into(),
                        ints: vec![0, 1], // Node 0 in tree 0, node 1 in tree 1
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_nodeids".into(),
                        ints: vec![0, 1],
                        ..Default::default()
                    },
                    // Minimal other attributes to pass parsing
                    AttributeProto {
                        name: "nodes_featureids".into(),
                        ints: vec![0, 0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_values".into(),
                        floats: vec![0.5, 0.0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_modes".into(),
                        strings: vec!["LEAF".into(), "LEAF".into()],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_truenodeids".into(),
                        ints: vec![0, 0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_falsenodeids".into(),
                        ints: vec![0, 0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_ids".into(),
                        ints: vec![0, 0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_weights".into(),
                        floats: vec![0.0, 0.0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_nodeids".into(),
                        ints: vec![0, 1],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_treeids".into(),
                        ints: vec![0, 1], // Different trees
                        ..Default::default()
                    },
                ],
            }],
        }),
        ..Default::default()
    };

    let onnx_bytes = model.encode_to_vec();
    let err = onnx::import_onnx(&onnx_bytes).unwrap_err();
    assert!(err.to_string().contains("Multi-tree ensembles"));
}

#[test]
fn test_reject_unsupported_node_mode() {
    use prost::Message;
    use zkml_prover::onnx::{
        AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorShapeProto,
        TensorShapeProtoDimension, TensorTypeProto, TypeProto, ValueInfoProto,
    };

    let model = ModelProto {
        ir_version: 8,
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
        graph: Some(GraphProto {
            name: "bad_mode".into(),
            input: vec![ValueInfoProto {
                name: "X".into(),
                r#type: Some(TypeProto {
                    tensor: Some(TensorTypeProto {
                        elem_type: 1,
                        shape: Some(TensorShapeProto {
                            dim: vec![TensorShapeProtoDimension {
                                dim_value: 1,
                                dim_param: String::new(),
                            }],
                        }),
                    }),
                }),
            }],
            output: vec![],
            node: vec![NodeProto {
                name: "bad_mode".into(),
                op_type: "TreeEnsembleClassifier".into(),
                domain: "ai.onnx.ml".into(),
                input: vec!["X".into()],
                output: vec!["Y".into()],
                attribute: vec![
                    AttributeProto {
                        name: "nodes_treeids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_nodeids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_featureids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_values".into(),
                        floats: vec![0.5],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_modes".into(),
                        strings: vec!["BRANCH_LT".into()], // Unsupported mode
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_truenodeids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_falsenodeids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_ids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_weights".into(),
                        floats: vec![0.0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_nodeids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_treeids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                ],
            }],
        }),
        ..Default::default()
    };

    let onnx_bytes = model.encode_to_vec();
    let err = onnx::import_onnx(&onnx_bytes).unwrap_err();
    assert!(err.to_string().contains("unsupported node mode"));
}

#[test]
fn test_reject_out_of_bounds_child_index() {
    use prost::Message;
    use zkml_prover::onnx::{
        AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorShapeProto,
        TensorShapeProtoDimension, TensorTypeProto, TypeProto, ValueInfoProto,
    };

    let model = ModelProto {
        ir_version: 8,
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
        graph: Some(GraphProto {
            name: "bad_index".into(),
            input: vec![ValueInfoProto {
                name: "X".into(),
                r#type: Some(TypeProto {
                    tensor: Some(TensorTypeProto {
                        elem_type: 1,
                        shape: Some(TensorShapeProto {
                            dim: vec![TensorShapeProtoDimension {
                                dim_value: 1,
                                dim_param: String::new(),
                            }],
                        }),
                    }),
                }),
            }],
            output: vec![],
            node: vec![NodeProto {
                name: "bad_index".into(),
                op_type: "TreeEnsembleClassifier".into(),
                domain: "ai.onnx.ml".into(),
                input: vec!["X".into()],
                output: vec!["Y".into()],
                attribute: vec![
                    AttributeProto {
                        name: "nodes_treeids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_nodeids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_featureids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_values".into(),
                        floats: vec![0.5],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_modes".into(),
                        strings: vec!["BRANCH_LEQ".into()],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_truenodeids".into(),
                        ints: vec![99], // Node ID 99 doesn't exist
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_falsenodeids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_ids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_weights".into(),
                        floats: vec![0.0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_nodeids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_treeids".into(),
                        ints: vec![0],
                        ..Default::default()
                    },
                ],
            }],
        }),
        ..Default::default()
    };

    let onnx_bytes = model.encode_to_vec();
    let err = onnx::import_onnx(&onnx_bytes).unwrap_err();
    assert!(err.to_string().contains("not found in nodes_nodeids"));
}

#[test]
fn test_reject_cycle_in_tree() {
    use prost::Message;
    use zkml_prover::onnx::{
        AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorShapeProto,
        TensorShapeProtoDimension, TensorTypeProto, TypeProto, ValueInfoProto,
    };

    // Create a cycle: node 0 -> node 1 -> node 0
    let model = ModelProto {
        ir_version: 8,
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
        graph: Some(GraphProto {
            name: "cycle".into(),
            input: vec![ValueInfoProto {
                name: "X".into(),
                r#type: Some(TypeProto {
                    tensor: Some(TensorTypeProto {
                        elem_type: 1,
                        shape: Some(TensorShapeProto {
                            dim: vec![TensorShapeProtoDimension {
                                dim_value: 2, // num_features=2 for cycle test
                                dim_param: String::new(),
                            }],
                        }),
                    }),
                }),
            }],
            output: vec![],
            node: vec![NodeProto {
                name: "cycle".into(),
                op_type: "TreeEnsembleClassifier".into(),
                domain: "ai.onnx.ml".into(),
                input: vec!["X".into()],
                output: vec!["Y".into()],
                attribute: vec![
                    AttributeProto {
                        name: "nodes_treeids".into(),
                        ints: vec![0, 0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_nodeids".into(),
                        ints: vec![0, 1],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_featureids".into(),
                        ints: vec![0, 1],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_values".into(),
                        floats: vec![0.5, 0.3],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_modes".into(),
                        strings: vec!["BRANCH_LEQ".into(), "BRANCH_LEQ".into()],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_truenodeids".into(),
                        ints: vec![1, 0], // Cycle: 0->1->0
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "nodes_falsenodeids".into(),
                        ints: vec![1, 1],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_ids".into(),
                        ints: vec![0, 0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_weights".into(),
                        floats: vec![0.0, 0.0],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_nodeids".into(),
                        ints: vec![0, 1],
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "class_treeids".into(),
                        ints: vec![0, 0],
                        ..Default::default()
                    },
                ],
            }],
        }),
        ..Default::default()
    };

    let onnx_bytes = model.encode_to_vec();
    let err = onnx::import_onnx(&onnx_bytes).unwrap_err();
    eprintln!("Cycle test error: {}", err);
    assert!(err.to_string().contains("cycle"));
}
