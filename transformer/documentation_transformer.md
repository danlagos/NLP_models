# Transformer Model Documentation

This document provides an overview and documentation for the `transformer.py` script, which implements a Transformer model using PyTorch. The Transformer model, introduced in "Attention is All You Need" by Vaswani et al., is a breakthrough in the field of natural language processing (NLP), providing a mechanism for sequence-to-sequence tasks without the use of recurrent layers.

## Modules and Libraries

The script utilizes the following PyTorch modules and libraries:

- `torch`
- `torch.nn`
- `torch.nn.functional` (implicitly through `nn` module operations)

## Classes Defined

### `SelfAttention`

**Purpose**: Implements the self-attention mechanism of the Transformer.

**Initialization Parameters**:
- `embed_size`: Size of the embedding vector.
- `heads`: Number of attention heads.

**Methods**:
- `forward(values, keys, query, mask)`: Computes the attention scores and applies them to the input.

### `TransformerBlock`

**Purpose**: Represents a single block in the Transformer, comprising self-attention and position-wise feed-forward network.

**Initialization Parameters**:
- `embed_size`, `heads`, `dropout`, `forward_expansion`.

**Methods**:
- `forward(value, key, query, mask)`: Processes input through the block, returning the transformed output.

### `Encoder`

**Purpose**: Encodes the input sequence using multiple `TransformerBlock`s.

**Initialization Parameters**:
- Parameters for managing embeddings and layer configurations, including `src_vocab_size`, `embed_size`, `num_layers`, etc.

**Methods**:
- `forward(x, mask)`: Encodes input `x` using self-attention and positional encoding.

### `DecoderBlock`

**Purpose**: A single block of the Decoder, handling self-attention on the decoder input and attention over the encoder output.

**Initialization Parameters**:
- Similar to `TransformerBlock`, adjusted for the decoder context.

**Methods**:
- `forward(x, value, key, src_mask, trg_mask)`: Processes decoder input and encoder output.

### `Decoder`

**Purpose**: Decodes the encoded input into the target sequence.

**Initialization Parameters**:
- Parameters for embeddings and layer configurations similar to `Encoder`.

**Methods**:
- `forward(x, enc_out, src_mask, trg_mask)`: Decodes the input sequence.

### `Transformer`

**Purpose**: Combines the Encoder and Decoder into a full Transformer model.

**Initialization Parameters**:
- Comprehensive parameters including vocab sizes, padding indices, and model configurations.

**Methods**:
- `forward(src, trg)`: Processes the entire sequence-to-sequence transformation.
- `make_src_mask`, `make_trg_mask`: Utility methods for mask generation.

### `TransformerDemo`

**Purpose**: Demonstrates the usage of the Transformer model.

**Initialization Parameters**:
- Model and device configurations.

**Methods**:
- `run_demo(x, trg)`: Runs a demonstration forward pass.

## Main Execution Block

Demonstrates the instantiation of the `TransformerDemo` class and runs a model demonstration with sample input and target data. It also checks for CUDA availability to use GPU acceleration if available.

## Usage

The script can be run directly to see a demonstration of the Transformer model in action. Ensure that PyTorch is installed and a suitable Python version is used as per PyTorch's requirements.

This documentation aims to provide a clear understanding of the structure and functionality of the `transformer.py` script, facilitating easy usage and modification for specific NLP tasks.
