//! Activation functions for quantized neural network layers.
//!
//! Only ZK-friendly activations are supported. ReLU is the workhorse because
//! it reduces to a comparison and a select, both cheap to constrain.

use crate::fixed_point::FixedPoint;

/// Quantized ReLU: `max(0, x)`.
///
/// Implemented as a signed comparison on the raw Q16.16 integer
/// (`if x.value < 0 { 0 } else { x }`), which mirrors the comparison plus
/// conditional-select gadget the circuit uses. This is the single shared
/// activation for both the native prover and the guest.
pub fn relu(x: FixedPoint) -> FixedPoint {
    if x.value < 0 {
        FixedPoint::from_raw(0, x.scale)
    } else {
        x
    }
}

/// Apply ReLU element-wise to a slice, returning a new vector.
pub fn relu_vec(xs: &[FixedPoint]) -> Vec<FixedPoint> {
    xs.iter().copied().map(relu).collect()
}

/// Quantized leaky ReLU: passes positives through and scales negatives by a
/// small slope instead of zeroing them.
///
/// The slope is expressed as a power-of-two right shift so it stays a cheap
/// constraint in the circuit (a shift, not a general multiply). `shift = 4`
/// approximates a slope of 1/16 (0.0625), within the common 0.01-0.1 range.
pub fn leaky_relu(x: FixedPoint, shift: u32) -> FixedPoint {
    if x.value < 0 {
        FixedPoint::from_raw(x.value >> shift, x.scale)
    } else {
        x
    }
}

/// Apply leaky ReLU element-wise to a slice, returning a new vector.
pub fn leaky_relu_vec(xs: &[FixedPoint], shift: u32) -> Vec<FixedPoint> {
    xs.iter().copied().map(|x| leaky_relu(x, shift)).collect()
}

/// Quantized ReLU6: `min(max(0, x), 6)`.
///
/// The upper bound keeps activations in a fixed range, which is common in
/// mobile-grade quantized networks and bounds the witness size in the circuit.
/// It reduces to two comparisons and a select, all ZK-friendly.
pub fn relu6(x: FixedPoint) -> FixedPoint {
    let zero = FixedPoint::from_raw(0, x.scale);
    let six = FixedPoint::from_raw(6i64 << x.scale, x.scale);
    x.clamp(zero, six)
}

/// Apply ReLU6 element-wise to a slice, returning a new vector.
pub fn relu6_vec(xs: &[FixedPoint]) -> Vec<FixedPoint> {
    xs.iter().copied().map(relu6).collect()
}

/// Quantized hard sigmoid: `relu6(x + 3) / 6`, an output in `[0, 1]`.
///
/// This is the piecewise-linear approximation of the logistic sigmoid used in
/// quantized mobile networks. Unlike the true sigmoid (an exponential, hostile
/// to ZK circuits) it is a shift, a clamp, and a constant division.
pub fn hard_sigmoid(x: FixedPoint) -> FixedPoint {
    let three = FixedPoint::from_raw(3i64 << x.scale, x.scale);
    let six = FixedPoint::from_raw(6i64 << x.scale, x.scale);
    let bounded = relu6(x.saturating_add(three));
    bounded
        .checked_div(six)
        .unwrap_or_else(|| FixedPoint::from_raw(0, x.scale))
}

/// Apply hard sigmoid element-wise to a slice, returning a new vector.
pub fn hard_sigmoid_vec(xs: &[FixedPoint]) -> Vec<FixedPoint> {
    xs.iter().copied().map(hard_sigmoid).collect()
}

/// Quantized hard swish: `x * hard_sigmoid(x)`.
///
/// A self-gated activation that keeps the smooth-ish shape of swish while
/// staying piecewise linear, so it remains practical to constrain in a circuit.
pub fn hard_swish(x: FixedPoint) -> FixedPoint {
    x * hard_sigmoid(x)
}

/// Apply hard swish element-wise to a slice, returning a new vector.
pub fn hard_swish_vec(xs: &[FixedPoint]) -> Vec<FixedPoint> {
    xs.iter().copied().map(hard_swish).collect()
}

/// Quantized hard tanh: `clamp(x, -1, 1)`.
///
/// The piecewise-linear stand-in for `tanh`. Like `relu6` it is two
/// comparisons and a select, so it stays cheap to constrain, but it is
/// symmetric around zero and saturates to `[-1, 1]`.
pub fn hardtanh(x: FixedPoint) -> FixedPoint {
    let one = FixedPoint::from_raw(1i64 << x.scale, x.scale);
    x.clamp(-one, one)
}

/// Apply hard tanh element-wise to a slice, returning a new vector.
pub fn hardtanh_vec(xs: &[FixedPoint]) -> Vec<FixedPoint> {
    xs.iter().copied().map(hardtanh).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_clamps_negatives() {
        let neg = FixedPoint::quantize(-2.0);
        assert_eq!(relu(neg).value, 0);
    }

    #[test]
    fn relu_passes_positives() {
        let pos = FixedPoint::quantize(2.0);
        assert_eq!(relu(pos).value, pos.value);
    }

    #[test]
    fn leaky_relu_keeps_positives() {
        let pos = FixedPoint::quantize(2.0);
        assert_eq!(leaky_relu(pos, 4).value, pos.value);
    }

    #[test]
    fn leaky_relu_dampens_negatives() {
        let neg = FixedPoint::quantize(-1.0);
        let out = leaky_relu(neg, 4);
        // Still negative, but closer to zero than the input.
        assert!(out.value < 0);
        assert!(out.value > neg.value);
    }

    #[test]
    fn leaky_relu_vec_applies_elementwise() {
        let xs = vec![FixedPoint::quantize(-1.0), FixedPoint::quantize(2.0)];
        let out = leaky_relu_vec(&xs, 4);
        assert_eq!(out.len(), 2);
        assert!(out[0].value < 0 && out[0].value > xs[0].value);
        assert_eq!(out[1].value, xs[1].value);
    }

    #[test]
    fn relu6_clamps_negatives_to_zero() {
        assert_eq!(relu6(FixedPoint::quantize(-2.0)).value, 0);
    }

    #[test]
    fn relu6_caps_at_six() {
        assert!((relu6(FixedPoint::quantize(9.0)).dequantize() - 6.0).abs() < 1e-4);
    }

    #[test]
    fn relu6_passes_mid_range() {
        assert!((relu6(FixedPoint::quantize(3.5)).dequantize() - 3.5).abs() < 1e-4);
    }

    #[test]
    fn hard_sigmoid_centers_at_half() {
        assert!((hard_sigmoid(FixedPoint::quantize(0.0)).dequantize() - 0.5).abs() < 1e-3);
    }

    #[test]
    fn hard_sigmoid_saturates() {
        assert!((hard_sigmoid(FixedPoint::quantize(10.0)).dequantize() - 1.0).abs() < 1e-3);
        assert!(hard_sigmoid(FixedPoint::quantize(-10.0)).dequantize().abs() < 1e-3);
    }

    #[test]
    fn hard_swish_zero_at_origin() {
        assert!(hard_swish(FixedPoint::quantize(0.0)).dequantize().abs() < 1e-3);
    }

    #[test]
    fn hard_swish_approaches_identity_for_large_positive() {
        // For x >= 3, hard_sigmoid saturates to 1, so hard_swish(x) == x.
        let x = FixedPoint::quantize(8.0);
        assert!((hard_swish(x).dequantize() - 8.0).abs() < 1e-2);
    }

    #[test]
    fn hard_swish_small_negative_is_negative() {
        let out = hard_swish(FixedPoint::quantize(-1.0));
        assert!(out.dequantize() < 0.0);
    }

    #[test]
    fn hardtanh_passes_mid_range() {
        assert!((hardtanh(FixedPoint::quantize(0.5)).dequantize() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn hardtanh_saturates_both_ends() {
        assert!((hardtanh(FixedPoint::quantize(4.0)).dequantize() - 1.0).abs() < 1e-4);
        assert!((hardtanh(FixedPoint::quantize(-4.0)).dequantize() + 1.0).abs() < 1e-4);
    }

    #[test]
    fn hardtanh_vec_applies_elementwise() {
        let xs = vec![FixedPoint::quantize(2.0), FixedPoint::quantize(-0.25)];
        let out = hardtanh_vec(&xs);
        assert_eq!(out.len(), 2);
        assert!((out[0].dequantize() - 1.0).abs() < 1e-4);
        assert!((out[1].dequantize() + 0.25).abs() < 1e-4);
    }

    #[test]
    fn relu6_vec_applies_elementwise() {
        let xs = vec![FixedPoint::quantize(-1.0), FixedPoint::quantize(9.0)];
        let out = relu6_vec(&xs);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].value, 0);
        assert!((out[1].dequantize() - 6.0).abs() < 1e-4);
    }

    #[test]
    fn hard_sigmoid_vec_applies_elementwise() {
        let xs = vec![FixedPoint::quantize(0.0), FixedPoint::quantize(10.0)];
        let out = hard_sigmoid_vec(&xs);
        assert_eq!(out.len(), 2);
        assert!((out[0].dequantize() - 0.5).abs() < 1e-3);
        assert!((out[1].dequantize() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn hard_swish_vec_applies_elementwise() {
        let xs = vec![FixedPoint::quantize(0.0), FixedPoint::quantize(8.0)];
        let out = hard_swish_vec(&xs);
        assert_eq!(out.len(), 2);
        assert!(out[0].dequantize().abs() < 1e-3);
        assert!((out[1].dequantize() - 8.0).abs() < 1e-2);
    }
}
