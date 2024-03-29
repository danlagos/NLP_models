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

# Self Attention Block
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
        
        # returns the linear transformation of the input        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
                
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

# start transformer block    
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
        self.dropout = nn.Dropout(dropout) # dropout
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask) # self attention
        x = self.dropout(self.norm1(attention + query)) # add & norm layer
        forward = self.feed_forward(x) # feed forward network
        out = self.dropout(self.norm2(forward + x)) # add & norm layer
        return out # output

# start encoder    
class Encoder(nn.Module):
    # this contains the hyperparameters of our model
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length, # max length of the sentence.  Used for positional encoding
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size) # embedding
        self.position_embedding = nn.Embedding(max_length, embed_size) # positional encoding
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
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions)) # add positional encoding
        )
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out

# start decoder block
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads=heads) # self attention
        self.norm = nn.LayerNorm(embed_size) # layer normalization
        self.transformer_block = TransformerBlock( # transformer block
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
    
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
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        out = self.fc_out(x) # prediction of what word is next
        return out
    
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx, # padding index, used for masking
        trg_pad_idx, # padding index, used for masking
        embed_size=256, # this might have to be adjusted due to GPU memory constraints
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda", # change to "cpu" if you don't have a GPU
        max_length=100,
    ):
        super(Transformer, self).__init__()
        
        # define encoder
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
        
        # define decoder
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
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        # (N, 1, 1, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
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

# example to see if this runs
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    
    out = model(x, trg[:, :-1])
    print(out.shape) # [N, trg_len - 1, trg_vocab_size]