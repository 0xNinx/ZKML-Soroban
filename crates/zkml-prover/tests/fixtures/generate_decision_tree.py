#!/usr/bin/env python3
"""
Generate test fixtures for decision tree ONNX import.

This script:
1. Trains a simple sklearn DecisionTreeClassifier
2. Exports it to ONNX using skl2onnx
3. Runs onnxruntime on sample inputs to get golden predictions
4. Saves the ONNX file and golden predictions as JSON

Usage:
    python generate_decision_tree.py
"""

import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# Generate a simple synthetic dataset
np.random.seed(42)
X_train = np.random.rand(100, 2)  # 100 samples, 2 features
y_train = (X_train[:, 0] + X_train[:, 1] > 1.0).astype(int)  # Simple threshold rule

# Train a decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Convert to ONNX
initial_type = [("float_input", FloatTensorType([None, 2]))]
onnx_model = convert_sklearn(tree, initial_types=initial_type)

# Save the ONNX model
onnx_path = "decision_tree_simple.onnx"
with open(onnx_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Saved ONNX model to {onnx_path}")

# Generate test samples and compute golden predictions using onnxruntime
test_samples = 20
X_test = np.random.rand(test_samples, 2).astype(np.float32)

# Run inference with onnxruntime
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

onnx_predictions = session.run([output_name], {input_name: X_test})[0]
onnx_classes = np.argmax(onnx_predictions, axis=1)

# Save golden predictions
golden_data = {
    "test_samples": X_test.tolist(),
    "predictions": onnx_classes.tolist(),
    "num_features": 2,
    "num_classes": 2,
}

golden_path = "decision_tree_simple_golden.json"
with open(golden_path, "w") as f:
    json.dump(golden_data, f, indent=2)

print(f"Saved golden predictions to {golden_path}")

print("\nFixture generation complete!")
print(f"  ONNX model: {onnx_path}")
print(f"  Golden predictions: {golden_path}")
print(f"  Test samples: {test_samples}")
print(f"  Features: {golden_data['num_features']}")
print(f"  Classes: {golden_data['num_classes']}")
