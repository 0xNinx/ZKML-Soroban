# Architecture

This document describes the system architecture of zkml-soroban, covering the
high-level design, component responsibilities, data flow, and the two
implementation routes planned for proof generation.

---

## Table of Contents

- [Design Principles](#design-principles)
- [High-Level Architecture](#high-level-architecture)
- [Component Breakdown](#component-breakdown)
  - [zkml-common](#zkml-common)
  - [zkml-prover](#zkml-prover)
  - [zkml-verifier](#zkml-verifier)
- [Data Flow](#data-flow)
- [Route A vs Route B](#route-a-vs-route-b)
- [On-Chain Storage Model](#on-chain-storage-model)
- [Security Considerations](#security-considerations)

---

## Design Principles

1. **Separation of proving and verification.** All computationally expensive
   operations (model inference, proof generation) happen off-chain. The
   on-chain contract performs only a single cryptographic check.

2. **Model-agnostic proof pipeline.** The prover is designed to support
   multiple model architectures through a shared interface. Adding a new model
   type requires implementing inference logic and, for Route B, a
   corresponding circuit.

3. **Fixed-point arithmetic everywhere.** ZK circuits operate over finite
   fields. All floating-point model parameters are quantized to fixed-point
   (Q16.16) at import time, ensuring deterministic computation across the
   prover and circuit.

4. **Minimal on-chain footprint.** The verifier contract stores only the
   model commitment hash, the most recent inference result, and the
   verification timestamp. Proof data is not persisted.

---

## High-Level Architecture

The system follows a standard off-chain prover / on-chain verifier pattern.

```
+------------------------------------------------------------------+
|                         Off-Chain                                 |
|                                                                   |
|  +------------+     +--------------+     +--------------------+   |
|  |            |     |              |     |                    |   |
|  | ONNX Model |---->| Quantization |---->| Inference Engine   |   |
|  |            |     | (f64 -> Q16) |     | (fixed-point eval) |   |
|  +------------+     +--------------+     +--------------------+   |
|                                                  |                |
|                                                  v                |
|                                          +---------------+        |
|                                          |               |        |
|                                          | Proof Engine  |        |
|                                          | (RISC Zero /  |        |
|                                          |  native circ) |        |
|                                          +-------+-------+        |
|                                                  |                |
+--------------------------------------------------+----------------+
                                                   |
                              Groth16 proof +      |
                              public inputs        |
                                                   v
+------------------------------------------------------------------+
|                         On-Chain (Soroban)                        |
|                                                                   |
|  +------------------------------------------------------------+  |
|  |                    zkml-verifier contract                   |  |
|  |                                                             |  |
|  |  1. Deserialize proof points (A, B, C)                     |  |
|  |  2. Reconstruct public input vector                        |  |
|  |  3. Call BN254 pairing check (CAP-0074 host functions)     |  |
|  |  4. If valid: record model_hash + output + ledger sequence |  |
|  +------------------------------------------------------------+  |
|                                                                   |
+------------------------------------------------------------------+
```

---

## Component Breakdown

### zkml-common

**Crate path:** `crates/zkml-common`

Shared library containing types and utilities used by both the prover and
verifier.

| Module        | Responsibility                                           |
|---------------|----------------------------------------------------------|
| `fixed_point` | Q16.16 fixed-point arithmetic with quantize/dequantize   |
| `models`      | Model representations: DecisionTree, LogisticRegression, TinyMLP |
| `proof`       | Groth16Proof, PublicInputs, VerificationBundle           |

This crate has no dependencies on Soroban SDK or RISC Zero, ensuring it can
be compiled for any target.

### zkml-prover

**Crate path:** `crates/zkml-prover`

Off-chain component responsible for the full inference-to-proof pipeline.

| Module         | Responsibility                                          |
|----------------|---------------------------------------------------------|
| `onnx`         | Parse ONNX protobuf files and extract model parameters  |
| `quantization` | Convert f64 weights and biases to FixedPoint            |
| `inference`    | Execute model inference using fixed-point arithmetic     |
| `prover`       | Generate Groth16 proofs via RISC Zero (Phase 1) or native circuits (Phase 2) |

**Inference engine details:**

- Decision tree: iterative node traversal comparing features against
  thresholds, all in fixed-point.
- Logistic regression: dot product of weight vector and input features plus
  bias. Sigmoid is omitted (not ZK-friendly); raw scores are compared against
  a threshold.
- Tiny MLP: sequential dense layer evaluation with quantized ReLU after
  every hidden layer (`max(0, x)` on the raw Q16.16 integer). The final
  layer emits raw linear scores; products use checked Q16.16 multiply.

### zkml-verifier

**Crate path:** `crates/zkml-verifier`

Soroban smart contract compiled to WASM and deployed to the Stellar network.

| Function            | Responsibility                                     |
|---------------------|----------------------------------------------------|
| `initialize`        | Store verification key and model commitment hash   |
| `verify_inference`  | Accept proof + public inputs, run Groth16 pairing check, record result |
| `get_result`        | Return the last verified inference record           |

The contract calls BN254 host functions (introduced by CAP-0074) to perform
the elliptic curve pairing check required by Groth16 verification:

```
e(A, B) == e(alpha, beta) * e(sum(pub_i * vk_i), gamma) * e(C, delta)
```

This is a single equation involving three pairings on the BN254 curve. The
host functions handle the expensive finite field and curve arithmetic
natively, keeping contract execution gas costs low.

---

## Data Flow

A complete inference verification cycle proceeds as follows:

1. **Model preparation** (one-time): The model owner exports a trained model
   to ONNX, imports it into zkml-prover, and quantizes all parameters.
   A Poseidon hash of the quantized parameters is computed and registered
   on-chain via `initialize`.

2. **Inference request**: A user submits input features to the prover service.

3. **Off-chain execution**:
   - The prover loads the quantized model and evaluates it on the inputs.
   - The inference runs inside the RISC Zero zkVM (Phase 1) or a native
     ZK circuit (Phase 2).
   - The proof system produces a Groth16 proof and the public inputs
     (model hash, input hash, output value).

4. **On-chain verification**: The prover submits the proof and public inputs
   to the `verify_inference` function. The contract:
   - Deserializes the proof curve points.
   - Calls the BN254 pairing host functions.
   - If valid, stores the `InferenceRecord` (model hash, output, ledger
     sequence).

5. **Result consumption**: Any on-chain or off-chain consumer can query
   `get_result` to retrieve the verified inference output and its timestamp.

---

## Route A vs Route B

### Route A: RISC Zero zkVM (Phase 1 -- MVP)

The inference logic is written as a standard Rust program and executed inside
the RISC Zero zkVM. The zkVM produces a STARK proof, which is then wrapped
into a Groth16 SNARK via the Bonsai proving service or a local prover.

**Advantages:**
- Rapid development: any Rust code can be proven.
- Battle-tested proving infrastructure.
- Already deployed on Soroban by Nethermind (`stellar-zk`).

**Trade-offs:**
- Generic zkVM overhead: proofs are larger and verification is more expensive
  than model-specific circuits.
- Prover latency is higher than specialized circuits.

### Route B: Native Circuits (Phase 2 -- Differentiation)

Custom ZK circuits are built specifically for each model architecture using
the BN254 and Poseidon primitives directly. For example, a decision tree
circuit encodes the tree structure as a sequence of conditional constraints.

**Advantages:**
- 10-50x smaller proofs compared to zkVM.
- Significantly cheaper on-chain verification.
- Circuit structure can exploit model regularity (e.g., balanced trees).

**Trade-offs:**
- Requires writing and auditing arithmetic circuits for each model type.
- Longer development timeline.

The planned sequence is to validate the concept with Route A and then
optimize with Route B, minimizing risk while maintaining a path to
production-grade efficiency.

---

## On-Chain Storage Model

The verifier contract uses Soroban instance storage for its state:

| Key          | Type              | Description                          |
|--------------|-------------------|--------------------------------------|
| `mdl_hash`   | `Bytes`           | Poseidon commitment to model params  |
| `lst_res`    | `InferenceRecord` | Last verified inference result       |
| `init`       | `bool`            | Initialization guard flag            |

Instance storage is appropriate here because the verification key and model
commitment are set once and referenced on every verification call. The
`InferenceRecord` is overwritten on each successful verification.

---

## Security Considerations

- **Proof soundness**: The security of the system relies on the soundness of
  the Groth16 proof system over the BN254 curve. A valid proof guarantees
  that the prover executed the claimed computation correctly.

- **Model binding**: The Poseidon hash commitment ensures that a proof is
  tied to a specific set of model parameters. Changing the model invalidates
  all existing proofs.

- **Input binding**: Public inputs include a hash of the input features,
  preventing the prover from substituting different inputs after generating
  the proof.

- **Contract access control**: The `initialize` function includes a
  re-entrancy guard to prevent re-initialization with a different model.
  Future iterations will add role-based access control for model registration.

- **Fixed-point overflow**: The quantization module must ensure that
  intermediate computations do not overflow the i64 range. This is verified
  by bounds analysis during model import.
