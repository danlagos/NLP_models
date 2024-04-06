import torch
import torch.nn as nn

""" 
The class initializes with the size of embeddings and the number of heads for multi-head attention. It ensures the embedding size is divisible by the number of heads. Inside, linear layers for values, keys, queries, and a final output are set up. The forward method reshapes inputs for multi-head processing, computes attention scores using scaled dot products (enhanced by a masking mechanism for padding), and combines the results through weighted sum and a final linear transformation
"""
# Self Attention Block
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        # Ensure the embedding size is divisible by the number of attention heads
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        # Initialize linear transformations for the input
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size) # Final output layer
    
    def forward(self, values, keys, query, mask):
        # Get the batch size and sequence lengths
        N = query.shape[0] # number of samples
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Reshape for multi-head attention, splitting the embedding size into 'heads' parts
        # This allows parallel computation over heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        # Apply linear transformation
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
                
        # Calculate the attention scores
        # 'energy' represents the compatibility between queries and keys
        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        
        # Apply masking if provided (for ignoring padding in the input sequences)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # Compute attention weights using softmax
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        # Apply attention weights to values
        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        # Pass through a final linear layer
        out = self.fc_out(out)
        return out

""" 
start transformer block 
This class defines a single transformer block, which includes self-attention followed by layer normalization, a feed-forward network, and another layer normalization step, all with dropout for regularization. The process integrates self-attention output with the original input (residual connection) before normalization, mirroring the transformer's characteristic path of flowing information
"""
# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads) # Initialize self-attention mechanism
        self.norm1 = nn.LayerNorm(embed_size) # First layer normalization
        self.norm2 = nn.LayerNorm(embed_size) # Second layer normalization
        
        # Initialize feed forward network
        # Expands the embedding size before reducing it back, adding capacity to the model
        self.feed_forward = nn.Sequential( 
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(), # Activation function for non-linearity
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout) # Dropout for regularization
        
    def forward(self, value, key, query, mask):
        # Compute self-attention
        attention = self.attention(value, key, query, mask) 
        # Apply dropout, then layer normalization (add & norm)
        x = self.dropout(self.norm1(attention + query)) 
        # Pass through feed forward network
        forward = self.feed_forward(x) 
        # Final dropout and normalization (add & norm)
        out = self.dropout(self.norm2(forward + x)) 
        return out # Return output
    
"""
start encoder    
This Encoder class is designed for the encoding phase of a transformer model. It creates embeddings for input tokens and applies positional encoding to retain information about the position of tokens in the input sequence. These embeddings are then processed through multiple transformer blocks, which apply self-attention and feed-forward networks to encode the input into a format useful for downstream tasks.
"""
# Start encoder
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,  # max length of the sentence for positional encoding
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # Embedding layer for input tokens
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        # Embedding layer for positional encoding
        self.position_embedding = nn.Embedding(max_length, embed_size)
        # Stack of Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Get batch size and sequence length
        N, seq_length = x.shape
        # Generate positional indices and move to device
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        # Apply dropout to the sum of token embeddings and positional encodings
        out = self.dropout(
            self.word_embedding(x) + self.position_embedding(positions)
        )
        
        # Pass through each Transformer layer
        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out


""" 
The DecoderBlock class is part of the transformer's decoder. It processes the target sequence input through self-attention and a transformer block, using both the source and target masks to appropriately limit the attention scope. This setup enables the model to focus on relevant input parts and prevent future token information from influencing the prediction of the current token.
"""
# Start decoder block
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        # Self-attention mechanism for the decoder's input
        self.attention = SelfAttention(embed_size, heads=heads)
        # Normalization layer for stabilizing the learning process
        self.norm = nn.LayerNorm(embed_size)
        # Transformer block for processing the output of the self-attention layer and the encoder's output
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        # Dropout for regularization to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, value, key, src_mask, trg_mask):
        # Self-attention with target mask (trg_mask) for preventing lookahead
        attention = self.attention(x, x, x, trg_mask)
        # Apply dropout and normalization to the sum of attention output and input
        query = self.dropout(self.norm(attention + x))
        # Process through transformer block with source mask (src_mask) to focus on relevant parts of the input
        out = self.transformer_block(value, key, query, src_mask)
        return out

""" 
The Decoder class is a component of the transformer model that decodes the encoded input into the target language sequence. It uses embeddings for the target vocabulary and positions, passes the sequence through multiple decoder blocks (each performing self-attention and encoder-decoder attention), and finally produces predictions for the next words in the sequence through a linear layer.
""" 
class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        # Embedding for target vocabulary
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        # Positional embedding for target sequence
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        # Decoder layers composed of multiple DecoderBlock instances
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        
        # Final linear layer for output predictions
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        # Get batch size and sequence length, then generate positional indices
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # Combine word and position embeddings with dropout
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        
        # Process through each decoder layer
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        # Apply the final linear layer to get predictions
        out = self.fc_out(x)
        return out
    
""" 
The Transformer class combines the encoder and decoder components, applying masks to handle padding and prevent lookahead. This design encapsulates the entire process of encoding the input, processing it through self-attention and feed-forward layers, and finally decoding it into the target sequence. Masks ensure the model treats padding appropriately and respects the sequential nature of language by preventing future information from influencing predictions.
"""

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,  # Index for source padding, used for mask creation
        trg_pad_idx,  # Index for target padding, used for mask creation
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",  # Can be set to "cpu" if no GPU is available
        max_length=100,
    ):
        super(Transformer, self).__init__()
        
        # Encoder part of the Transformer
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )
        
        # Decoder part of the Transformer
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )
        
        # Padding indexes for source and target
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        # Create a mask for the source input to ignore padding
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        # Create a mask for the target: prevent the model from looking ahead
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

""" 
The TransformerDemo class encapsulates the initialization and demonstration of a Transformer model for NLP tasks. It takes vocabulary sizes for source and target languages, padding indices, and device information (CPU/GPU) as inputs to set up the model. The class provides a run_demo method that executes a forward pass of the model with sample input and target data, showcasing how to prepare data, run predictions, and interpret the model's output dimensions. This class serves as a practical example of using the Transformer in a PyTorch-based workflow.
"""

# Define a class for demonstrating the Transformer model
class TransformerDemo:
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device):
        # Initialize the Transformer model with given parameters and assign it to the specified device (GPU/CPU)
        self.model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
        self.device = device

    # Function to run a demo prediction using the Transformer model
    def run_demo(self, x, trg):
        # Execute the model forward pass and trim the last token from the target for prediction purposes
        out = self.model(x, trg[:, :-1])
        # Print the shape of the output to check the model's prediction dimensions
        print(out.shape)

""" 
This code block serves as the main entry point for executing a demonstration of the Transformer model when the script is run directly. It begins by checking for the availability of a CUDA-capable GPU to utilize hardware acceleration for computations, defaulting to the CPU if a GPU is not available. It then instantiates the TransformerDemo class, which encapsulates the setup and operation of the Transformer model, including its initialization with specified vocabulary sizes and padding indices. The script prepares input (x) and target (trg) tensors, which are randomly generated sequences of integers representing encoded text, and transfers them to the appropriate computational device (GPU or CPU). Finally, it executes the model's forward pass using these tensors to demonstrate the Transformer's predictive capabilities. The output's shape is printed to verify the model's functionality and to illustrate the dimensions of the predicted sequences.
"""

# This block runs if the script is executed as the main program
if __name__ == "__main__":
    # Determine if a CUDA capable GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create an instance of the demo class with specified parameters
    demo = TransformerDemo(10, 10, 0, 0, device)
    # Prepare input and target tensors, moving them to the chosen device
    x = torch.tensor([[6, 3, 7, 4, 6, 9, 2, 6, 7], [4, 3, 7, 7, 2, 5, 4, 1, 7]]).to(device)
    trg = torch.tensor([[5, 1, 4, 0, 9, 5, 8, 0], [9, 2, 6, 3, 8, 2, 4, 2]]).to(device)
    # Run the demo with the prepared data
    demo.run_demo(x, trg)
