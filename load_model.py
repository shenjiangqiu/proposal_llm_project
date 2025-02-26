# %%
from enum import Enum

from transformers import LlamaForCausalLM, GPT2LMHeadModel


class ModelType(Enum):
    Llama = 1
    GPT2 = 2


def load_model(model_type: ModelType):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    queries = {}
    keys = {}
    values = {}

    def capture_qkv(module, input, output):
        # Queries, keys, and values from the attention layer input
        queries["data"] = input[
            0
        ].detach()  # Shape: (batch_size, seq_length, hidden_dim)
        keys["data"] = module.key(input[0]).detach()
        values["data"] = module.value(input[0]).detach()
        pass

    if model_type == ModelType.Llama:
        name = "meta-llama/Llama-3.2-3B-Instruct"
        model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(name)
        print("Llama model loaded")

    elif model_type == ModelType.GPT2:
        name = "openai-community/gpt2"
        model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(name)
        model.transformer.h[0].attn.register_forward_hook(capture_qkv)
        print("GPT2 model loaded")

    tokenizer = AutoTokenizer.from_pretrained(name)

    return model, tokenizer, queries, keys, values
    pass


# %%
if __name__ == "__main__":
    model, tokenizer, queries, keys, values = load_model(ModelType.Llama)
    prompt = """
    Question: How to install pytorch in linux, return me the pure text with out any format. Answer:
    To install PyTorch in Linux, follow these steps:

1. Update your package list:  
   ```bash
   sudo apt update
   ```

2. Install Python and pip if not already installed:  
   ```bash
   sudo apt install python3 python3-pip
   ```

3. Install PyTorch using pip:  
   ```bash
   pip3 install torch torchvision torchaudio
   ```

4. If you need GPU support with CUDA, use the official installation command from PyTorch's website:  
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   Replace `cu121` with the appropriate CUDA version.

5. Verify installation:  
   ```bash
   python3 -c "import torch; print(torch.__version__)"
   ```
"""

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")
    model(**inputs, save_info=True)
    load_model(ModelType.GPT2)
    pass

# %%
