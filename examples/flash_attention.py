import torch
import torch.backends.cuda
import torch.nn.functional as F

torch.set_default_device("cuda")
torch.backends.cuda.enable_flash_sdp(enabled=True)

B = 1
N = 32768
E = 1024

q = torch.randn(B, N, E)
k = torch.randn(B, N, E)
v = torch.randn(B, N, E)

o = F.scaled_dot_product_attention(q, k, v)
print(o)

