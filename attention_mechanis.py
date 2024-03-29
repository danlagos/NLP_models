"""
code to attention mechanism.  

Reverse engineered peer reviewed article "Attention is all you need"

encoder:
receives input
create emedding
create positional encoding
add positional encoding to embedding

Encoder Attention Block
    creat multi-head attention
        input is split into multiple heads.  1 for Q, 1 for K, and 1 for V, 1 for add & norm layer.  this block only accepts input for Q, K, and V.
        output is concatenated
    add & norm layer
        take input from multi-head attention and add it to the input
        output is split in to 2 parts.  1 for add and norm layer and 1 for feed forward network
    feed forward network
        takes input from add & norm layer
        output is passed to a add & norm layer
    add & norm layer
        take input from feed forward layer, and the previous add & norm layer
        output
end of encoder attention block 

decoder:

output fron encoder is split into two, and is passed to a multi-head attention block in the decoder.
primary difference between decoder and encoder
    decoder has a masked multi-head attention block.
    a multi-head attention block that takes the output from the encoder (2 channels) and the output from the masked multi-head attention block.
    
receives input
create emedding
create positional encoding
add positional encoding to embedding
Decoder Attention Block
    Masked Multi-head attention
        input is split into multiple heads.  1 for Q, 1 for K, and 1 for V, 1 for add & norm layer.  this block only accepts input for Q, K, and V.
        output is concatenated
    add & norm layer
        input comes from masked multi-head attention and the input
        output is split into 2 parts.  1 for add & norm layer and 1 for another add & norm layer
    Multi-head attention
        input comes from encoder, 2 channels, and the output from the add & norm layer.  total of 3 channels
        output is concatenated
    add & norm layer
        input comes from multi-head attention and the previous add & norm layer
        output is split into 2 parts.  1 for add & norm layer and 1 for feed forward network
    feed forward network
        input comes from add & norm layer
        output is passed to a add & norm layer
    add & norm layer
        input comes from feed forward network and the previous add & norm layer
        output is passed to the next block
    end of decoder attention block
    
    linear layer
        input comes from the add & norm layer
        output is passed to the softmax function
    softmax function
        input comes from the linear layer
        output is the prediction
        
notes on multi-head attention:
    takes embeddig inputs, say 256 dimensions, and split into 8 parts.  32 dimensions each.
    each part is passed through a linear layer, through a scaled dot-prodcut Attention.
        scaled dot-prodcut Attention: Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    this output is concatenated and passed through another linear layer.
"""

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size) # fully connected layer
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0] # number of samples
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim) # (N, value_len, heads, head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim) # (N, key_len, heads, head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim) # (N, query_len, heads, head_dim)
        
        """
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        """
        
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)

        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # out shape: (N, query_len, heads, head_dim)
        # we want to return (N, query_len, heads, head_dim), so we need to transpose and reshape
        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        # fc_out matches embed_size to embed_size
        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads) # self attention
        self.norm1 = nn.LayerNorm(embed_size) # layer normalization
        self.norm2 = nn.LayerNorm(embed_size) # layer normalization
        
        # feed forward network
        # input is embed_size, output is embed_size
        # input is passed through a linear layer, then a ReLU activation function, then another linear layer
        self.feed_forward = nn.Sequential( 
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)