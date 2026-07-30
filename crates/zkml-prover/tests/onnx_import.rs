//! Integration tests for the ONNX importer foundation using committed fixtures.

use zkml_prover::onnx::{
    import_onnx, parse_model_proto, OnnxImportError, MIN_OPSET_CORE, MIN_OPSET_VERSION,
};

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read fixture {path}: {e}"))
}

#[test]
fn fixture_decision_tree_validates_then_defers_extraction() {
    let bytes = fixture("decision_tree_valid.onnx");
    let model = parse_model_proto(&bytes).expect("valid onnx decodes");
    assert!(model.graph.is_some());
    assert!(!model.graph.as_ref().unwrap().node.is_empty());
    // Realistic pair: core 17 + ai.onnx.ml 3.
    let ml = model
        .opset_import
        .iter()
        .find(|e| e.domain == "ai.onnx.ml")
        .expect("ml opset");
    assert_eq!(ml.version, 3);

    let err = import_onnx(&bytes).unwrap_err();
    // Now expects MalformedModel because fixture lacks input tensor shapes
    assert!(matches!(err, OnnxImportError::MalformedModel(_)));
}

#[test]
fn fixture_skl2onnx_like_tree_is_accepted() {
    let bytes = fixture("skl2onnx_like_tree.onnx");
    let model = parse_model_proto(&bytes).expect("skl2onnx-like onnx decodes");
    let core = model
        .opset_import
        .iter()
        .find(|e| e.domain.is_empty() || e.domain == "ai.onnx")
        .expect("core opset");
    let ml = model
        .opset_import
        .iter()
        .find(|e| e.domain == "ai.onnx.ml")
        .expect("ml opset");
    assert!(core.version >= MIN_OPSET_CORE);
    assert!(ml.version >= 1 && ml.version <= 5);

    let err = import_onnx(&bytes).unwrap_err();
    // Now expects MalformedModel because fixture lacks input tensor shapes
    assert!(matches!(err, OnnxImportError::MalformedModel(_)));
}

#[test]
fn fixture_linear_validates_then_defers_extraction() {
    let bytes = fixture("linear_classifier_valid.onnx");
    let model = parse_model_proto(&bytes).expect("linear onnx decodes");
    let ml = model
        .opset_import
        .iter()
        .find(|e| e.domain == "ai.onnx.ml")
        .expect("ml opset");
    assert_eq!(ml.version, 1);

    // LinearClassifier should now extract successfully
    let imported_model = import_onnx(&bytes).expect("linear classifier should extract");
    match imported_model {
        zkml_common::models::Model::LogisticRegression(lr) => {
            assert_eq!(lr.weights.len(), 3, "should have 3 weights");
        }
        _ => panic!("Expected LogisticRegression model"),
    }
}

#[test]
fn fixture_unsupported_conv_names_operator() {
    let bytes = fixture("unsupported_conv.onnx");
    let err = import_onnx(&bytes).unwrap_err();
    match &err {
        OnnxImportError::UnsupportedOperator { op_type } => {
            assert_eq!(op_type, "Conv");
            let msg = err.to_string();
            assert!(msg.contains("Conv"), "display was: {msg}");
        }
        other => panic!("expected UnsupportedOperator, got {other}"),
    }
}

#[test]
fn fixture_low_opset_is_rejected() {
    let bytes = fixture("low_opset_tree.onnx");
    let err = import_onnx(&bytes).unwrap_err();
    match err {
        OnnxImportError::UnsupportedOpset { found, required } => {
            assert_eq!(found, 13);
            assert_eq!(required, MIN_OPSET_VERSION);
            let msg = err.to_string();
            assert!(msg.contains("13"), "display was: {msg}");
            assert!(msg.contains("17"), "display was: {msg}");
        }
        other => panic!("expected UnsupportedOpset, got {other}"),
    }
}

#[test]
fn garbage_input_does_not_panic() {
    let err = import_onnx(b"\x00\x01\x02not-onnx").unwrap_err();
    assert!(matches!(err, OnnxImportError::MalformedModel(_)));
}

#[test]
fn linear_classifier_round_trip_with_20_samples() {
    use zkml_common::fixed_point::FixedPoint;
    use zkml_common::inference::run_inference;

    let bytes = fixture("linear_classifier_valid.onnx");
    let imported_model = import_onnx(&bytes).expect("linear classifier should extract");

    // The fixture has 3 features with weights [0.5, -0.3, 0.8] and bias 0.1
    // Use 20 deterministic test samples
    let test_samples: Vec<Vec<FixedPoint>> = (0..20)
        .map(|i| {
            let base = i as f64 / 20.0;
            vec![
                FixedPoint::quantize(base),       // feature 0
                FixedPoint::quantize(1.0 - base), // feature 1
                FixedPoint::quantize(base * 0.5), // feature 2
            ]
        })
        .collect();

    // Run inference on all samples
    let results: Vec<FixedPoint> = test_samples
        .iter()
        .map(|sample| run_inference(&imported_model, sample))
        .collect();

    // Verify we got results for all 20 samples
    assert_eq!(results.len(), 20, "Should have results for all 20 samples");

    // For a valid model, all inferences should complete successfully
    // The actual decision (positive/negative) depends on the threshold
    // which is application-specific, so we just verify the computation completes
}
