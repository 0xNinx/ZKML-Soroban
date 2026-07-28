# zkml-verifier

On-chain Soroban smart contract for ZK proof verification.

## Requirements

### Minimum Protocol Version
- **Stellar Protocol 25 (X-Ray)** or higher
- BN254 host functions (CAP-0074) are required for Groth16 verification

### Minimum SDK Version
- **soroban-sdk 27.0.0** or higher
- The contract uses BN254 cryptographic operations introduced in SDK v25.0.0

### Build Target
- **wasm32v1-none** (not wasm32-unknown-unknown)
- Rust 1.84+ is required for the wasm32v1-none target
- For Rust 1.81 or earlier, use wasm32-unknown-unknown (not recommended for SDK v27+)

## Contract Size

The release build produces a WASM binary of approximately **20KB**, well within Soroban's size limits.

## Usage

### Initialization
The contract must be initialized with a model commitment and Groth16 verification key:

```rust
initialize(model_hash: Bytes, vk: VerificationKey)
```

### Verification
Verify a Groth16 proof with public inputs:

```rust
verify_inference(
    proof_a: Bytes,      // 64 bytes - G1 point
    proof_b: Bytes,      // 128 bytes - G2 point
    proof_c: Bytes,      // 64 bytes - G1 point
    public_inputs: Bytes // model_hash (32) || input_hash (32) || output
) -> Result<(), VerificationError>
```

### Public Inputs Format
- `model_hash`: 32-byte Poseidon commitment to model parameters
- `input_hash`: 32-byte Poseidon commitment to input features
- `output`: Variable-length inference output

## Verification Equation

The contract implements the Groth16 verification equation:

```
e(A, B) == e(alpha, beta) * e(L, gamma) * e(C, delta)
```

Where `L = sum(public_input_i * vk_ic_i)` is computed using BN254 scalar multiplication and point addition.

## Error Codes

| Error Code | Description |
|------------|-------------|
| 1 | ContractNotInitialized |
| 2 | PublicInputsTooShort |
| 3 | MalformedProofA |
| 4 | MalformedProofB |
| 5 | MalformedProofC |
| 6 | MalformedVerificationKey |
| 7 | VerificationFailed |

## Security Considerations

- The contract only records results after successful pairing verification
- Model hash binding prevents proof replay across different models
- No unwrap/expect on the verification path - all errors are typed
- Verification key is stored once and cannot be changed after initialization
