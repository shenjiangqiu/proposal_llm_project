# %%
import torch
import os
import struct
import numpy as np


# Print the results
# %%
tensor_result = {}
for i in range(28):
    file_name = f"qk_internal_{i}.pt"
    if os.path.exists(file_name):
        qk_internal = torch.load(file_name)
        print(f"qk_internal_{i}.pt shape: {qk_internal.shape}")
        tensor_result = qk_internal.detach().numpy()
        pass
# %%
print(tensor_result)
# %%


def reduce(x):
    # print(x) : array of {"sign": sign, "exponent": exponent}
    print(x.shape)

    # Convert to float32 and sum
    top_sum = np.sum(x, axis=-1)

    # print(f"Sum of top 32 values: {top_sum}")
    return top_sum


reduced = np.apply_along_axis(reduce, -1, tensor_result)
sorted_token_indices = np.argsort(reduced, axis=-1)
print(sorted_token_indices)
print(reduced.shape)

# %%
# save the sorted token indices
np.save("use_full_sorted_token_indices.npy", sorted_token_indices)

# %%
