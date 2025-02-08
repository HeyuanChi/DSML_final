import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, 
                 input_dim=3, 
                 d_model=512, 
                 nhead=8, 
                 num_encoder_layers=4, 
                 num_decoder_layers=4, 
                 dim_feedforward=2048, 
                 dropout=0.1, 
                 output_dim=3, 
                 max_len=100000):
        super(TimeSeriesTransformer, self).__init__()
        
        # Input embedding
        self.encoder_input_layer = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

        # Output embedding
        self.decoder_input_layer = nn.Linear(input_dim, d_model)
        self.pos_decoder = PositionalEncoding(d_model, max_len=max_len)
        
        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Linear
        self.linear_layer = nn.Linear(d_model, output_dim)
        
    def forward(self, src, tgt):
        """
        src: [batch_size, src_len, input_dim]
        tgt: [batch_size, tgt_len, input_dim]
        """
        src_emb = self.encoder_input_layer(src)         # Embedding -> [batch_size, src_len, d_model]
        src_emb = self.pos_encoder(src_emb)             # Positional Encoding
        memory = self.encoder(src_emb)                  # Encoding  -> [batch_size, src_len, d_model]
        
        tgt_emb = self.decoder_input_layer(tgt)         # Embedding -> [batch_size, tgt_len, d_model]
        tgt_emb = self.pos_decoder(tgt_emb)             # Positional Encoding
        
        # Mask: [tgt_len, tgt_len]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(1)).to(tgt_emb.device)
        
        out = self.decoder(tgt=tgt_emb, 
                           memory=memory, 
                           tgt_mask=tgt_mask)           # Decoding -> [batch_size, tgt_len, d_model]
        
        pred = self.linear_layer(out)                   # Linear   -> [batch_size, tgt_len, output_dim]
        return pred
