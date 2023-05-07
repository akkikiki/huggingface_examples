import torch
import torch.backends.cuda
import torch.nn.functional as F

torch.set_default_device("cuda")
torch.backends.cuda.enable_flash_sdp(enabled=True)

B = 1  # batch size
N = 32768  # context length
E = 1024  # embedding dim

q = torch.randn(B, N, E)
k = torch.randn(B, N, E)
v = torch.randn(B, N, E)

o = F.scaled_dot_product_attention(q, k, v)
print(o)
print(o.size())

