# %%
import torch

a = torch.zeros((2,3,12))
b = torch.zeros((2,12,4))
print(a.shape)
# %%
c=  torch.unsqueeze(a, 1)
print(c.shape)
# %%
r = a.matmul(b)
print(r.shape)
# %%
ele = a.unsqueeze(-1) * b.unsqueeze(1)
# %%
print(ele.shape)
# %%
