import torch.nn as nn

class ImageTokenizer(nn.Module):
    """
    simplified one layer tokenizer based on CCT
    architecture
    """
    def __init__(self,
                 kernel_size,stride,padding,
                 pooling_kernel_size=3,pooling_stride=2,pooling_padding=1,
                 n_input_channels=3,n_output_channels=768,bias=False
                ):
        super().__init__()
        self.conv_layer = nn.Conv2d(n_input_channels,
                                    n_output_channels,
                                    stride=stride,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    bias=bias)
        self.pool_layer = nn.Sequential(nn.MaxPool2d( kernel_size=pooling_kernel_size,
                                        stride=pooling_stride,
                                        padding=pooling_padding
                                        ),
                                        nn.GELU(),
                                        nn.Dropout(0.3)
                                        )
        self.flattener = nn.Flatten(2)

        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m,(nn.Conv2d)):
            nn.init.orthogonal_(m.weight)



    def forward(self,x):
        """
        turns an image B x 3 x H x W into B x N x D
        """
        return self.flattener( self.pool_layer( self.conv_layer(x) ) ).transpose(-2,-1)