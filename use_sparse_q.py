# %%
import torch
import os
import struct
import numpy as np


def float_to_ieee754(num):
    """Convert a float32 number to IEEE 754 representation (binary string)."""
    # Pack float into 4 bytes and unpack as an integer
    packed = struct.pack(">f", num)  # Convert float to bytes
    int_rep = struct.unpack(">I", packed)[0]  # Convert bytes to unsigned int
    # Convert to binary and format as 32 bits
    return f"{int_rep:032b}"


num = -24
binary_rep = float_to_ieee754(num)
print(f"IEEE 754 representation of {num}: {binary_rep}")


def ieee754_breakdown(binary):
    """Breaks down IEEE 754 binary representation into sign, exponent, and mantissa."""
    sign = int(binary[0], 2)
    exponent = int(binary[1:9], 2) - 127  # Exponent with bias removed
    mantissa = 1  # Start with implicit 1

    # Convert mantissa binary fraction to decimal
    for i, bit in enumerate(binary[9:], 1):
        mantissa += int(bit) * (2**-i)

    return sign, exponent, mantissa


sign, exponent, mantissa = ieee754_breakdown(binary_rep)
print(f"Sign: {sign}, Exponent: {exponent}, Mantissa: {mantissa}")


def extract_sign_exponent(f):
    """Extracts sign and exponent from a float32 number."""
    # Convert float to IEEE 754 binary representation
    bits = struct.unpack(">I", struct.pack(">f", f))[0]

    sign = (bits >> 31) & 0x1  # Extract sign bit
    exponent = ((bits >> 23) & 0xFF) - 127  # Extract exponent and remove bias

    return {"sign": sign, "exponent": exponent}


def tensor_to_struct_array(tensor):
    """Maps a torch tensor into an array of structs containing sign and exponent."""
    tensor_np = tensor.detach().numpy()  # Convert to NumPy for easy iteration
    return np.vectorize(extract_sign_exponent)(tensor_np)


tensor = torch.tensor([[12.0, -7.5], [0.1, 256.0]], dtype=torch.float32)

# Convert to structured array
result = tensor_to_struct_array(tensor)

# Print the results
print(result)
# %%
tensor_result = {}
for i in range(28):
    file_name = f"qk_internal_{i}.pt"
    if os.path.exists(file_name):
        qk_internal = torch.load(file_name)
        print(f"qk_internal_{i}.pt shape: {qk_internal.shape}")
        tensor_result = tensor_to_struct_array(qk_internal)
        pass
# %%
print(tensor_result)
# %%


def reduce_struct(x):
    # print(x) : array of {"sign": sign, "exponent": exponent}
    signs = np.array([i["sign"] for i in x])
    exponents = np.array([i["exponent"] for i in x])
    top_32_indices = np.argsort(exponents)[-32:]
    top_exponents = exponents[top_32_indices]
    top_signs = signs[top_32_indices]
    top_signs = np.where(top_signs == 0, 1, -1)

    # Print results
    top_values = top_signs * (2.0**top_exponents)

    # Convert to float32 and sum
    top_sum = np.sum(top_values.astype(np.float32))

    # print(f"Sum of top 32 values: {top_sum}")
    return top_sum


reduced = np.apply_along_axis(reduce_struct, -1, tensor_result)
sorted_token_indices = np.argsort(reduced, axis=-1)
print(sorted_token_indices)
print(reduced.shape)

# %%
# save the sorted token indices
np.save("use_e_sorted_token_indices.npy", sorted_token_indices)

# %%
