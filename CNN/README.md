# CNN using torch

```python
import torch.nn

torch.nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    bias=True,
    padding_mode='zeros'
)
```
### group
- group=1, 모든 입력=>모든 출력과 convolution. 일반적인 convolution 연산과 같음.   
- groups=2, 입력을 2그룹으로 나눠 각각 convolution 연산을 수행하고 그 결과를 concatenation   
- groups=in_channels, 각각의 input_channel이 각각의 output_channel에 대응되어 convolution 연산을 수행, size= out_channels // in_channels   

