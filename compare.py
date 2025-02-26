# %%
# compare use_e and use_f
import numpy as np

use_e = np.load("use_e_sorted_token_indices.npy")
use_f = np.load("use_full_sorted_token_indices.npy")
use_efull = np.load("use_efull_sorted_token_indices.npy")
print(use_e.shape)
print(use_f.shape)

# %%
top_8_e = use_e[:, -8:]
top_8_f = use_f[:, -8:]
top_8_efull = use_efull[:, -8:]

top_16_e = use_e[:, -16:]
top_16_f = use_f[:, -16:]
top_16_efull = use_efull[:, -16:]

top32_e = use_e[:, -32:]
top32_f = use_f[:, -32:]
top32_efull = use_efull[:, -32:]
# %%
recalls8of32 = np.array(
    [len(set(top_8_e[i]) & set(top32_f[i])) / 8.0 for i in range(24)]
)
print("recalls8of32")
print(recalls8of32)
# %%

recalls8of8 = np.array(
    [len(set(top_8_e[i]) & set(top_8_f[i])) / 8.0 for i in range(24)]
)
print("recalls8of8")
print(recalls8of8)
# %%

recallefull8of32 = np.array(
    [len(set(top_8_efull[i]) & set(top32_f[i])) / 8.0 for i in range(24)]
)
print("recallefull8of32")
print(recallefull8of32)
# %%

recallefull8of16 = np.array(
    [len(set(top_8_efull[i]) & set(top_16_f[i])) / 8.0 for i in range(24)]
)
print("recallefull8of16")
print(recallefull8of16)
# %%
recallsefull8of8 = np.array(
    [len(set(top_8_efull[i]) & set(top_8_f[i])) / 8.0 for i in range(24)]
)
print("recallsefull8of8")
print(recallsefull8of8)
# %%
recalls16of32 = np.array(
    [len(set(top_16_e[i]) & set(top32_f[i])) / 16.0 for i in range(24)]
)
print(recalls16of32)
# %%
recalls32of32 = np.array(
    [len(set(top32_e[i]) & set(top32_f[i])) / 32.0 for i in range(24)]
)
print(recalls32of32)

# %%
