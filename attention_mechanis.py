    """
    code to attention mechanism.  
    Reverse engineering the peer reviewed article "Attention is all you need"
    
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
    """

