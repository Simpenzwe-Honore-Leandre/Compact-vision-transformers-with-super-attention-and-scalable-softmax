import torch
import torch.nn as nn
import torch.nn.functional as F
from super_mha import SuperMultiHeadAttention
from tokenizer import ImageTokenizer

class SuperCCTransformer(nn.Module):
  """
  SCCTransformer: Super attention Compact Convolutional Transformer.
  Uses conv tokenization,super attention blocks, and learnable pooling.
  Args:
      input_shape: tuple (B,C,H,W)
      kernel_size, stride, padding: for tokenizer conv
      embed_dim: token embedding dimension
      ...
  Returns:
      Tensor of shape (B, 1)
  """
  def __init__(self,
               input_shape,
               kernel_size,
               stride,
               padding,
               embed_dim=768,
               pooling_kernel_size=3,
               pooling_stride=2,
               pooling_padding=1,
               num_heads   = 24,
               num_layers = 6,
               bias =False
               ):
    super().__init__()
    B,C,H,W = input_shape
    self.tokenizer = ImageTokenizer(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        pooling_kernel_size=pooling_kernel_size,
        pooling_stride=pooling_stride,
        pooling_padding=pooling_padding,
        n_input_channels=C,
        n_output_channels=embed_dim,
        bias=bias
    )
    with torch.no_grad():
      dummy_input = torch.randn(1, *input_shape[1:])
      _, self.seq_len, _ = self.tokenizer(dummy_input).shape
    self.head_dim    = embed_dim//num_heads
    self.embed_dim   = embed_dim
    self.num_heads   = num_heads
    self.num_layers  = num_layers

    self.super_attn_blocks = nn.Sequential( *[ SuperMultiHeadAttention(
                                                    seq_len=self.seq_len,
                                                    num_heads=self.num_heads,
                                                    embed_dim=self.embed_dim,
                                                    bias=False)
                                                   for _ in range( num_layers )]  )
    self.pooler = SeqPooler(in_features=self.embed_dim,out_features=1,bias=bias)

    self.apply( self.init_weight )

  @staticmethod
  def init_weight(m):
    if isinstance(m,(nn.Conv2d,nn.Linear)):
      nn.init.orthogonal_(m.weight)
      if m.bias is not None:
                nn.init.zeros_(m.bias)



  def forward(self,x):
    return self.pooler(self.super_attn_blocks( self.tokenizer(x) ) )


class SeqPooler(nn.Module):
  """
  Attention-based sequence pooler.
  Computes a weighted average of sequence tokens using learned attention scores.

  Input: (B, N, D)
  Output: (B, D)
  """

  def __init__(self,
               in_features,
               out_features,
               bias=False
               ):
    super().__init__()
    self.attention_pool  = nn.Linear(in_features=in_features, out_features=1,bias=bias)
    self.dropout = nn.Dropout(0.3)
  def forward(self,x):
    pooled_seq=  torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
    return self.dropout(pooled_seq)
