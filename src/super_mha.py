import torch.nn as nn
import torch
import torch.nn.functional as F

class SuperMultiHeadAttention(nn.Module):
    """
    wide attention layer to prevent rank collapse,
    ssmax,
    super attention,efficient attention,optimised attention,standard,
    differential attetniton,
    skip connections and pre-norm
    Adam8bit optim, ademamix,grokfast??
    """
    def __init__(
        self,
        seq_len,
        scale_param,# should be in transformer module since shared across all layers
        num_heads=12,
        embed_dim = 768,
        bias = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim  = self.embed_dim // self.num_heads
        self.scale_param = scale_param.view(1, num_heads, 1, 1)

        self.attn_scaler = self.head_dim**-0.5
        self.logn_scaler = torch.log( torch.tensor( self.embed_dim , dtype=torch.float32))

        self.norm1 = nn.RMSNorm(embed_dim)
        self.norm2 = nn.RMSNorm(embed_dim)


        self.ff = nn.Sequential(
            nn.Linear(self.embed_dim,self.embed_dim,bias=bias),
            nn.GELU(),
            nn.Linear(self.embed_dim,self.embed_dim,bias=bias)
        )

        self.q_proj= nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
            bias=bias,
        )

        self.v_proj = nn.Linear(
            in_features=seq_len,
            out_features=seq_len,
            bias=bias,
        )

        self.WA = nn.Parameter( torch.empty(seq_len,seq_len) )
        nn.init.orthogonal_(self.WA)


    def forward(self, x)->torch.Tensor :
        B,N,C = x.shape

        x_norm= self.norm1(x)

        q = self.q_proj(x_norm).reshape(B, self.num_heads,N, self.head_dim)

        #slicing
        k = x_norm.reshape(B, self.num_heads,N, self.head_dim).permute(0, 1, 3, 2).contiguous()  # transposed by default

        v = x_norm.reshape(B,self.num_heads , N, self.head_dim)

        wa = self.WA[:N, :N]

        v = wa @ v

        # scalable softmax for attention applied to q vector
        scores = F.softmax( self.logn_scaler * self.scale_param *  q @ k,dim=-1 )

        attn = (scores @ v).reshape(B,N,C) + x

        return  self.ff( self.norm2(attn) )  + attn



