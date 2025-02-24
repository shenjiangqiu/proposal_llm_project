# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Define a dictionary to store the queries
queries = {}

# Define a hook function to capture the queries from the attention layer
def capture_queries(module, input, output):
    # Queries are in input[0] for GPT2Attention
    queries['data'] = input[0].detach()

# Attach the hook to the first layer's attention mechanism
hook = model.transformer.h[0].attn.register_forward_hook(capture_queries)

# Define a prompt
prompt = "Arch Linux defines simplicity as without unnecessary additions or modifications. It ships software as released by the original developers (upstream) with minimal distribution-specific (downstream) changes: patches not accepted by upstream are avoided, and Arch's downstream patches consist almost entirely of backported bug fixes that are obsoleted by the project's next release.In a similar fashion, Arch ships the configuration files provided by upstream with changes limited to distribution-specific issues like adjusting the system file paths. It does not add automation features such as enabling a service simply because the package was installed. Packages are only split when compelling advantages exist, such as to save disk space in particularly bad cases of waste. GUI configuration utilities are not officially provided, encouraging users to perform most system configuration from the shell and a text editor. "


# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt")

# Forward pass to capture queries
with torch.no_grad():
    outputs = model(**inputs)

# Remove the hook after capturing
hook.remove()

# Extract and inspect the query vector for a specific token (e.g., last token)
token_index = -3 # Index for the last token
query_vector = queries['data'][0, token_index]  # Shape: (num_heads * head_dim)

# Print the query vector shape and values
print(f"Query Vector Shape: {query_vector.shape}")
print(f"Query Vector for Token '{tokenizer.decode(inputs['input_ids'][0, token_index])}':\n{query_vector}")

# Optionally, save the query vector
torch.save(query_vector, "query_vector.pt")
print("\nQuery vector saved to query_vector.pt")

# %%
