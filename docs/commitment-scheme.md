# Poseidon Commitment Scheme

This document specifies the consensus-critical commitment scheme used for model and input binding in zkml-soroban. The scheme uses Poseidon hash over the BN254 scalar field, matching the Soroban host functions from CAP-0075.

## Overview

The commitment scheme provides two cryptographic bindings:

1. **Model commitment**: A Poseidon hash of all quantized model parameters serves as the on-chain model identifier. This ensures a proof for model M cannot validate for model M'.
2. **Input commitment**: A Poseidon hash of the input feature vector is included as a public input, preventing the prover from swapping inputs after proving.

Both commitments use the same Poseidon parameters to ensure interoperability between off-chain prover computation and on-chain verification.

## Field and Parameters

### Field

- **Field**: BN254 Fr (scalar field of the BN254 elliptic curve)
- **Field order**: `r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001`
- **Host function parameter**: `field = 1` (BN254 Fr)

### Poseidon Parameters

The commitment scheme uses Poseidon with parameters matching the circomlib implementation for BN254, as provided by the `rs-soroban-poseidon` SDK:

- **State size (t)**: 3
- **Rate (r)**: 2 (number of input elements per permutation)
- **Capacity (c)**: 1 (domain separation element)
- **S-box degree (d)**: 5
- **Rounds full (rounds_f)**: 8 (must be even)
- **Rounds partial (rounds_p)**: 57
- **MDS matrix**: Standard Cauchy matrix from circomlib
- **Round constants**: Pre-generated constants from circomlib

These parameters are the default BN254 configuration in `rs-soroban-poseidon` and match the widely-used circomlib implementation, ensuring interoperability with existing ZK tooling.

### Domain Separation

Domain separation prevents cross-contamination between different commitment types:

- **Model commitment**: Capacity element initialized to `1` (domain tag for models)
- **Input commitment**: Capacity element initialized to `2` (domain tag for inputs)

The capacity element is set before absorbing any data and remains unchanged throughout the sponge computation.

## Serialization

### Model Parameter Serialization

Model parameters are serialized into field elements in the following order:

#### Logistic Regression

1. Weights: `weights[0].value, weights[1].value, ..., weights[n-1].value` (little-endian i64)
2. Bias: `bias.value` (little-endian i64)

#### Decision Tree

1. Number of features: `num_features` (as i64)
2. For each node in order (index 0 to n-1):
   - **Split node**: `feature_index` (as i64), `threshold.value` (little-endian i64), `left` (as i64), `right` (as i64)
   - **Leaf node**: `value.value` (little-endian i64)

#### TinyMLP

For each layer in order:
1. Weights: `weights[0].value, weights[1].value, ..., weights[m-1].value` (row-major order, little-endian i64)
2. Biases: `biases[0].value, biases[1].value, ..., biases[k-1].value` (little-endian i64)
3. Input size: `input_size` (as i64)
4. Output size: `output_size` (as i64)

All FixedPoint values are serialized as their raw i64 representation (the quantized integer value, not the dequantized float).

### Input Feature Serialization

Input features are serialized as a flat vector of FixedPoint values:

1. `features[0].value, features[1].value, ..., features[n-1].value` (little-endian i64)

The order matches the feature index order expected by the model.

## Sponge Construction

The commitment uses the standard Poseidon sponge construction:

### Absorption Phase

1. Initialize state with capacity element (domain tag) and zeros:
   - `state[0] = domain_tag` (1 for model, 2 for input)
   - `state[1] = 0`
   - `state[2] = 0`

2. For each input element (from serialization):
   - If rate is full (2 elements available), absorb both:
     - `state[1] = element_i`
     - `state[2] = element_{i+1}`
     - Apply Poseidon permutation
   - If only one element remains, absorb it and pad with zero:
     - `state[1] = element_last`
     - `state[2] = 0`
     - Apply Poseidon permutation

### Squeeze Phase

1. After final permutation, extract the capacity element as the hash:
   - `hash = state[0]`

2. Convert the field element to 32 bytes (little-endian)

### Multi-Round Absorption

For inputs exceeding the rate (more than 2 elements), the sponge performs multiple absorption rounds:

```
state = [domain_tag, 0, 0]
for chunk in input.chunks(2) {
    state[1] = chunk[0]
    state[2] = chunk.get(1).unwrap_or(&0)
    state = poseidon_permutation(state)
}
hash = state[0]
```

## Field Element Conversion

### i64 to BN254 Fr

FixedPoint values are stored as i64 integers. To convert to BN254 Fr field elements:

1. Take the absolute value of the i64
2. Convert to u64
3. Interpret as a field element (since i64 max value < BN254 field order)
4. For negative values, use the field's modular inverse (or store as signed integer and handle in circuit)

**Implementation note**: The current implementation uses direct little-endian byte encoding of the i64 value, which is then interpreted as a field element. This works because the absolute value of any i64 fits within the BN254 field.

### BN254 Fr to Bytes

The output field element is converted to 32 bytes using little-endian encoding:

```
bytes = field_element.to_bytes_le()  // 32 bytes
```

## Off-Chain Implementation

The off-chain implementation uses the `rs-soroban-poseidon` crate:

```rust
use soroban_poseidon::poseidon_hash;
use soroban_sdk::{crypto::bn254::Bn254Fr, vec, Env, U256};

// For model commitment (domain tag = 1)
let env = Env::default();
let elements: Vec<U256> = model_elements(model)
    .into_iter()
    .map(|v| U256::from_u64(&env, v.unsigned_abs()))
    .collect();
let hash = poseidon_hash::<3, Bn254Fr>(&env, &elements);
```

**Note**: The off-chain implementation must use the same parameters as the on-chain host function. The `rs-soroban-poseidon` crate provides the exact circomlib-compatible parameters for BN254.

## On-Chain Usage

The verifier contract uses the Poseidon host function from CAP-0075:

```rust
use soroban_sdk::Env;

// In verify_inference:
let stored_model_hash = env.storage().instance().get(&MODEL_HASH);
let public_model_hash = public_inputs.slice(0..32);

// Verify that the proof's model_hash matches the stored commitment
assert_eq!(stored_model_hash, public_model_hash);
```

The contract does not recompute the hash on-chain (to avoid gas costs). Instead, it verifies that the model_hash public input matches the pre-computed commitment stored during initialization.

## Security Properties

### Collision Resistance

The Poseidon hash function with the specified parameters provides collision resistance up to the 128-bit security level of BN254. Any change to the serialized parameters will produce a different hash with overwhelming probability.

### Binding

- **Model binding**: The model commitment is stored on-chain during initialization. A proof for model M will only verify if the public input contains the hash of M. Changing any model parameter changes the hash, breaking the proof.
- **Input binding**: The input commitment is included in the public inputs. The prover cannot substitute a different input after generating the proof without invalidating the hash.

### Uniqueness

The domain separation ensures that model commitments and input commitments live in distinct hash spaces, preventing cross-type collisions.

## Test Vectors

Test vectors are provided in the `zkml-common` crate to ensure off-chain and on-chain implementations produce identical digests. See `crates/zkml-common/tests/commitment_cross_check.rs` for the cross-check test.

## References

- CAP-0075: Poseidon hash host functions (Stellar Protocol 25)
- rs-soroban-poseidon: https://github.com/stellar/rs-soroban-poseidon
- circomlib Poseidon implementation: https://github.com/iden3/circomlib/blob/master/circuits/poseidon.circom
- Poseidon paper: "Poseidon: A New Hash Function for Zero-Knowledge Proof Systems" (Grassi et al., 2021)
