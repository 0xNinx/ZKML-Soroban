//! Acceptance tests for TinyMLP fixed-point inference (issue #7).

use zkml_common::error::ZkmlError;
use zkml_common::fixed_point::FixedPoint;
use zkml_common::models::{DenseLayer, Model, TinyMLP};
use zkml_prover::inference::{run_inference, try_run_inference};

fn fp(x: f64) -> FixedPoint {
    FixedPoint::quantize(x)
}

/// Hand-designed 2→2→1 network used by several tests.
///
/// Layer 1 (2→2): W = [[1, 0], [0, 1]], b = [0, -1]
/// Layer 2 (2→1): W = [[0.5, 1]], b = [0.25]
///
/// For input [2.0, 0.5]:
///   pre-ReLU hidden = [2.0, -0.5] → after ReLU [2.0, 0.0]
///   raw score = 0.5*2 + 1*0 + 0.25 = 1.25
///   Q16.16 raw = 81920
fn network_2_2_1() -> TinyMLP {
    TinyMLP {
        layers: vec![
            DenseLayer {
                weights: vec![fp(1.0), fp(0.0), fp(0.0), fp(1.0)],
                biases: vec![fp(0.0), fp(-1.0)],
                input_size: 2,
                output_size: 2,
            },
            DenseLayer {
                weights: vec![fp(0.5), fp(1.0)],
                biases: vec![fp(0.25)],
                input_size: 2,
                output_size: 1,
            },
        ],
    }
}

#[test]
fn hand_computed_2_2_1_matches_exact_q16_16() {
    let model = Model::TinyMLP(network_2_2_1());
    let out = run_inference(&model, &[fp(2.0), fp(0.5)]);
    // Exact Q16.16 for 1.25: round(1.25 * 2^16) = 81920
    assert_eq!(out.value, 81920);
    assert_eq!(out.scale, 16);
}

#[test]
fn relu_zeroes_negative_hidden_activations() {
    // Without ReLU the second hidden unit stays -0.5 and the final score
    // would be 0.75 (raw 49152). With ReLU it is zeroed → final 1.25.
    let model = Model::TinyMLP(network_2_2_1());
    let out = run_inference(&model, &[fp(2.0), fp(0.5)]);
    assert_eq!(out.value, 81920);
    assert_ne!(out.value, 49152);
}

#[test]
fn shape_mismatch_input_length_returns_error() {
    let model = Model::TinyMLP(network_2_2_1());
    let err = try_run_inference(&model, &[fp(1.0)])
        .expect_err("single feature must not match input_size 2");
    assert_eq!(
        err,
        ZkmlError::FeatureCountMismatch {
            expected: 2,
            got: 1
        }
    );
}

#[test]
fn shape_mismatch_layer_chain_returns_error() {
    let mlp = TinyMLP {
        layers: vec![
            DenseLayer {
                weights: vec![fp(1.0), fp(0.0), fp(0.0), fp(1.0)],
                biases: vec![fp(0.0), fp(0.0)],
                input_size: 2,
                output_size: 2,
            },
            // Deliberately expects 3 inputs while previous layer outputs 2.
            DenseLayer {
                weights: vec![fp(1.0); 3],
                biases: vec![fp(0.0)],
                input_size: 3,
                output_size: 1,
            },
        ],
    };
    assert!(matches!(mlp.validate(), Err(ZkmlError::InvalidModel(_))));
    let model = Model::TinyMLP(mlp);
    let err = try_run_inference(&model, &[fp(1.0), fp(2.0)])
        .expect_err("unchained layers must be rejected before arithmetic");
    assert!(matches!(err, ZkmlError::InvalidModel(_)));
}

/// Maximum allowed absolute deviation between the fixed-point output and the
/// f64 reference for the golden network.
///
/// Bound rationale: each Q16.16 quantize or multiply-shift introduces at most
/// about 1 ULP (`1 / 2^16` ~= 1.5e-5) of absolute error; across two dense
/// layers the accumulated error stays well under the documented ceiling.
const GOLDEN_ABS_ERR_BOUND: f64 = 1e-4;

/// Golden network chosen so the fixed-point path actually rounds. The weights
/// (0.3, -0.2, 0.1, 0.4, 0.6, 0.2) are not exact multiples of `2^-16`, and the
/// negative weight drives a negative product whose arithmetic shift floors
/// toward negative infinity. That is the case most likely to diverge between a
/// naive float model and the proven fixed-point path, so it is the one worth
/// pinning with a golden reference.
fn golden_network() -> TinyMLP {
    TinyMLP {
        layers: vec![
            DenseLayer {
                weights: vec![fp(0.3), fp(-0.2), fp(0.1), fp(0.4)],
                biases: vec![fp(0.1), fp(-0.5)],
                input_size: 2,
                output_size: 2,
            },
            DenseLayer {
                weights: vec![fp(0.6), fp(0.2)],
                biases: vec![fp(0.05)],
                input_size: 2,
                output_size: 1,
            },
        ],
    }
}

fn float_golden(inputs: &[f64; 2]) -> f64 {
    // Same weights as `golden_network`, evaluated in f64. ReLU on the hidden
    // layer, raw score on the output.
    let h0 = 0.3 * inputs[0] + (-0.2) * inputs[1] + 0.1;
    let h1 = 0.1 * inputs[0] + 0.4 * inputs[1] + (-0.5);
    let a0 = h0.max(0.0);
    let a1 = h1.max(0.0);
    0.6 * a0 + 0.2 * a1 + 0.05
}

#[test]
fn golden_float_reference_within_documented_bound() {
    let model = Model::TinyMLP(golden_network());
    let inputs_f = [0.5_f64, 0.5];
    let fixed_out = run_inference(&model, &[fp(inputs_f[0]), fp(inputs_f[1])]).dequantize();
    let float_out = float_golden(&inputs_f);
    let err = (fixed_out - float_out).abs();
    // The network is chosen so quantization actually rounds: the deviation must
    // be nonzero (otherwise the bound proves nothing) yet within the ceiling.
    assert!(
        err > 0.0,
        "expected a nonzero rounding deviation, got fixed={fixed_out} float={float_out}"
    );
    assert!(
        err <= GOLDEN_ABS_ERR_BOUND,
        "fixed={fixed_out} float={float_out} abs_err={err} exceeds bound {GOLDEN_ABS_ERR_BOUND}"
    );
}
