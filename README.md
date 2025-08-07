Repo implementing compact convolutional transformers with scalable softmax.

First install dependencies from requirements.txt file



To use the tokenizer.

```python
from src.tokenizer import ImageTokenizer
tokenizer =  ImageTokenizer(3,3,0,n_input_channels=3)
```

To use the super multihead attention blocks

```python
from src.super_mha import SuperMultiHeadAttention
mha = SuperMultiHeadAttention(seq_len=25,
                              num_heads=12,
                              scale_param=nn.Parameter(torch.ones(12)),
                              embed_dim=768,
                              bias=False
                              )
```

To use the whole sscct transformer block

```python
from src.sscct_transformer import SSCCTransformer
model = SSCCTransformer(images.shape,kernel_size=3,stride=3,padding=0,embed_dim=768,num_layers=6)
```
