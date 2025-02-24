# %%
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Define dictionaries to store queries, keys, and values
queries = {}
keys = {}
values = {}

# Hook functions to capture Q, K, V from the first attention block
def capture_qkv(module, input, output):
    # Queries, keys, and values from the attention layer input
    queries['data'] = input[0].detach()  # Shape: (batch_size, seq_length, hidden_dim)
    keys['data'] = module.key(input[0]).detach()
    values['data'] = module.value(input[0]).detach()

# Attach hook to the first layer's attention
hook = model.transformer.h[0].attn.register_forward_hook(capture_qkv)

# Define a prompt
prompt = "In the future, AI will"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Forward pass to capture Q, K, V
with torch.no_grad():
    outputs = model(**inputs, use_cache=True, return_dict_in_generate=True)

# Remove the hook
hook.remove()

# Extract Q, K, V from the captured data
q = queries['data']  # Shape: (batch_size, seq_length, hidden_dim)
k = keys['data']     # Shape: (batch_size, seq_length, hidden_dim)
v = values['data']   # Shape: (batch_size, seq_length, hidden_dim)

# Select the query of the last token
last_token_q = q[0, -1, :]  # Shape: (hidden_dim,)

# Compute Q * K^T for the last token
qk_product = torch.matmul(k[0], last_token_q)  # Shape: (seq_length,)

# Visualize Q * K^T
plt.figure(figsize=(10, 5))
plt.plot(qk_product.cpu().numpy())
plt.title("Q * K^T Attention Scores (First Block, Last Token)")
plt.xlabel("Key Token Position")
plt.ylabel("Dot Product Score")
plt.grid(True)
plt.show()

# Save Q, K, V, and QK^T for further inspection
torch.save(last_token_q, "last_token_query.pt")
torch.save(k, "first_block_keys.pt")
torch.save(v, "first_block_values.pt")
torch.save(qk_product, "qk_product.pt")

print("Saved:")
print("- Last token query vector (Q) to 'last_token_query.pt'")
print("- All keys (K) to 'first_block_keys.pt'")
print("- All values (V) to 'first_block_values.pt'")
print("- Q * K^T result to 'qk_product.pt'")

# %%
