//! Minimal ONNX protobuf schema used by the importer foundation.
//!
//! Field numbers match the official ONNX `onnx.proto` definitions so that
//! real `.onnx` files decode correctly for the messages we care about
//! (model metadata, opset imports, graph nodes).
//!
//! Only the surface required for parse + validation is modelled. Attribute
//! and tensor payloads needed for parameter extraction will be added with
//! issues #5 / #6.
//!
//! Schema reference: ONNX IR (opset 17+ era), `ModelProto` / `GraphProto` /
//! `NodeProto` / `OperatorSetIdProto`.

use prost::Message;

/// Top-level ONNX model container (`ModelProto`).
#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    /// IR version of the model encoding.
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    /// Human-readable producer name (optional).
    #[prost(string, tag = "2")]
    pub producer_name: String,
    /// Producer version string (optional).
    #[prost(string, tag = "3")]
    pub producer_version: String,
    /// Domain name for the model (optional).
    #[prost(string, tag = "4")]
    pub domain: String,
    /// Model version number (optional).
    #[prost(int64, tag = "5")]
    pub model_version: i64,
    /// Free-form documentation string.
    #[prost(string, tag = "6")]
    pub doc_string: String,
    /// Computation graph.
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    /// Operator set imports declaring domain + version pairs.
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
}

/// Operator set identifier (`OperatorSetIdProto`).
#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetIdProto {
    /// Domain name. Empty string means the default ONNX domain (`ai.onnx`).
    #[prost(string, tag = "1")]
    pub domain: String,
    /// Opset version for this domain.
    #[prost(int64, tag = "2")]
    pub version: i64,
}

/// Computation graph (`GraphProto`).
#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    /// Ordered list of operator nodes.
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    /// Graph name.
    #[prost(string, tag = "2")]
    pub name: String,
}

/// Single operator node (`NodeProto`).
#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    /// Input tensor names.
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    /// Output tensor names.
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    /// Optional node name.
    #[prost(string, tag = "3")]
    pub name: String,
    /// Operator type (e.g. `MatMul`, `TreeEnsembleClassifier`).
    #[prost(string, tag = "4")]
    pub op_type: String,
    /// Operator domain. Empty means default ONNX domain.
    #[prost(string, tag = "7")]
    pub domain: String,
    /// Node attributes (e.g. coefficients, intercepts for LinearClassifier).
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
}

/// Attribute definition (`AttributeProto`).
///
/// Used to store operator parameters such as weights, biases, and configuration.
#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    /// Attribute name.
    #[prost(string, tag = "1")]
    pub name: String,
    /// Single float value.
    #[prost(float, tag = "2")]
    pub f: f32,
    /// List of float values.
    #[prost(float, repeated, tag = "11")]
    pub floats: Vec<f32>,
    /// Single int64 value.
    #[prost(int64, tag = "3")]
    pub i: i64,
    /// List of int64 values.
    #[prost(int64, repeated, tag = "12")]
    pub ints: Vec<i64>,
    /// Single string value (as bytes).
    #[prost(bytes, tag = "4")]
    pub s: Vec<u8>,
    /// List of string values (as bytes).
    #[prost(bytes, repeated, tag = "13")]
    pub strings: Vec<Vec<u8>>,
    /// Tensor value (for weights/biases stored as tensors).
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    /// Graph value (for nested graphs).
    #[prost(message, optional, tag = "6")]
    pub g: Option<GraphProto>,
    /// Sparse tensor value.
    #[prost(message, optional, tag = "14")]
    pub sparse_tensor: Option<SparseTensorProto>,
    /// Type identifier (proto enum).
    #[prost(int32, tag = "20")]
    pub r#type: i32,
}

/// Tensor representation (`TensorProto`).
///
/// Used for storing multi-dimensional arrays like weight matrices.
#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    /// Data type enum (e.g. FLOAT=1, INT32=6).
    #[prost(int32, tag = "1")]
    pub data_type: i32,
    /// Shape dimensions.
    #[prost(int64, repeated, tag = "2")]
    pub dims: Vec<i64>,
    /// Raw data buffer (for float32 data).
    #[prost(bytes, tag = "9")]
    pub raw_data: Vec<u8>,
    /// Float data (if not using raw_data).
    #[prost(float, repeated, tag = "5")]
    pub float_data: Vec<f32>,
    /// Int32 data (if not using raw_data).
    #[prost(int32, repeated, tag = "6")]
    pub int32_data: Vec<i32>,
    /// String data.
    #[prost(string, repeated, tag = "8")]
    pub string_data: Vec<String>,
    /// Int64 data.
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,
    /// Tensor name.
    #[prost(string, tag = "3")]
    pub name: String,
}

/// Sparse tensor representation (`SparseTensorProto`).
///
/// Placeholder for future sparse tensor support.
#[derive(Clone, PartialEq, Message)]
pub struct SparseTensorProto {
    /// Dense tensor representing values.
    #[prost(message, optional, tag = "1")]
    pub values: Option<TensorProto>,
}
