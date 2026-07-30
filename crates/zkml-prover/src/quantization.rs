//! Model quantization utilities.
//!
//! Converts floating-point model parameters into fixed-point
//! representations suitable for arithmetic inside ZK circuits.

use zkml_common::fixed_point::FixedPoint;

/// Quantize a vector of `f64` values into fixed-point representations.
pub fn quantize_weights(weights: &[f64]) -> Vec<FixedPoint> {
    weights.iter().copied().map(FixedPoint::quantize).collect()
}

/// Quantize a single bias value.
pub fn quantize_bias(bias: f64) -> FixedPoint {
    FixedPoint::quantize(bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_preserves_sign() {
        let positive = quantize_bias(1.5);
        let negative = quantize_bias(-1.5);
        assert!(positive.value > 0);
        assert!(negative.value < 0);
    }

    #[test]
    fn quantize_vector() {
        let weights = vec![0.1, 0.2, 0.3];
        let quantized = quantize_weights(&weights);
        assert_eq!(quantized.len(), 3);
        for (orig, q) in weights.iter().zip(quantized.iter()) {
            assert!((orig - q.dequantize()).abs() < 1e-4);
        }
    }
}

/// Maximum absolute reconstruction error introduced by quantizing `values`.
///
/// Useful for asserting that a model survives quantization within tolerance.
pub fn max_quantization_error(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|&v| (v - FixedPoint::quantize(v).dequantize()).abs())
        .fold(0.0, f64::max)
}

/// Dequantize a slice of fixed-point values back to `f64`.
pub fn dequantize_all(values: &[FixedPoint]) -> Vec<f64> {
    values.iter().map(FixedPoint::dequantize).collect()
}

#[cfg(test)]
mod tests_error {
    use super::*;

    #[test]
    fn error_within_tolerance() {
        let values = vec![0.1, -0.25, std::f64::consts::PI, 100.5, -42.0];
        assert!(max_quantization_error(&values) < 1e-4);
    }
}

/// Min-max scale a feature vector into the `[0, 1]` range before quantization.
///
/// A constant vector maps to all zeros (no information to preserve).
pub fn scale_features(raw: &[f64]) -> Vec<f64> {
    let min = raw.iter().copied().fold(f64::INFINITY, f64::min);
    let max = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    if range == 0.0 {
        return raw.iter().map(|_| 0.0).collect();
    }
    raw.iter().map(|&x| (x - min) / range).collect()
}

#[cfg(test)]
mod tests_scaling {
    use super::*;

    #[test]
    fn scaling_preserves_order() {
        let raw = vec![10.0, -5.0, 3.0, 20.0];
        let scaled = scale_features(&raw);
        // The smallest raw value should map to the smallest scaled value.
        let min_idx = 1;
        let max_idx = 3;
        assert!((scaled[min_idx] - 0.0).abs() < 1e-9);
        assert!((scaled[max_idx] - 1.0).abs() < 1e-9);
    }
}

/// A summary of the error introduced by quantizing a parameter set.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationReport {
    /// Number of values quantized.
    pub count: usize,
    /// Maximum absolute reconstruction error.
    pub max_error: f64,
}

/// Build a [`QuantizationReport`] for a set of floating-point values.
pub fn quantization_report(values: &[f64]) -> QuantizationReport {
    QuantizationReport {
        count: values.len(),
        max_error: max_quantization_error(values),
    }
}

#[cfg(test)]
mod tests_report {
    use super::*;

    #[test]
    fn report_within_tolerance() {
        let report = quantization_report(&[0.1, 0.2, 0.3, -0.4]);
        assert_eq!(report.count, 4);
        assert!(report.max_error < 1e-4);
    }
}

/// Test that quantized weights deviate from the float originals by < 2⁻¹⁶ each.
#[cfg(test)]
mod tests_quantization_accuracy {
    use super::*;

    #[test]
    fn quantization_error_less_than_2_minus_16() {
        // 2⁻¹⁶ ≈ 1.5259e-5
        const MAX_ALLOWED_ERROR: f64 = 1.0 / 65536.0; // 2⁻¹⁶

        let test_values = vec![
            0.0, 1.0, -1.0, 0.5, -0.5, 0.1, -0.1, 0.25, -0.25, 100.0, -100.0, 0.01, -0.01, 3.14159,
            -3.14159,
        ];

        for &value in &test_values {
            let quantized = FixedPoint::quantize(value);
            let dequantized = quantized.dequantize();
            let error = (value - dequantized).abs();

            assert!(
                error < MAX_ALLOWED_ERROR,
                "Quantization error {} for value {} exceeds 2⁻¹⁶ ({})",
                error,
                value,
                MAX_ALLOWED_ERROR
            );
        }
    }

    #[test]
    fn weight_vector_quantization_accuracy() {
        const MAX_ALLOWED_ERROR: f64 = 1.0 / 65536.0; // 2⁻¹⁶

        let weights = vec![0.5, -0.3, 0.8, 0.1, -0.2];
        let quantized = quantize_weights(&weights);

        for (orig, q) in weights.iter().zip(quantized.iter()) {
            let dequantized = q.dequantize();
            let error = (orig - dequantized).abs();

            assert!(
                error < MAX_ALLOWED_ERROR,
                "Weight quantization error {} for original {} exceeds 2⁻¹⁶",
                error,
                orig
            );
        }
    }

    #[test]
    fn bias_quantization_accuracy() {
        const MAX_ALLOWED_ERROR: f64 = 1.0 / 65536.0; // 2⁻¹⁶

        let bias = 0.12345;
        let quantized = quantize_bias(bias);
        let dequantized = quantized.dequantize();
        let error = (bias - dequantized).abs();

        assert!(
            error < MAX_ALLOWED_ERROR,
            "Bias quantization error {} exceeds 2⁻¹⁶",
            error
        );
    }
}
