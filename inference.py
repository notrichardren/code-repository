import os
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download, snapshot_download
import torch

MODEL_NAME = f"meta-llama/Llama-2-70b-hf"
WEIGHTS_DIR = f"{os.getcwd()}/llama-weights-70b"


# Download model

if not os.path.exists(WEIGHTS_DIR):
    os.system(f"mkdir {WEIGHTS_DIR}")

checkpoint_location = snapshot_download(model_name, local_dir=WEIGHTS_DIR, ignore_patterns=["*.safetensors", "model.safetensors.index.json"]) # run this if you haven't downloaded the 70b model
checkpoint_location = WEIGHTS_DIR # run this if you haven't


# Load model

with init_empty_weights():
    model = LlamaForCausalLM.from_pretrained(checkpoint_location)

model = load_checkpoint_and_dispatch(
    model,
    checkpoint_location,
    device_map="auto",
    offload_folder=WEIGHTS_DIR,
    dtype=torch.float16,
    no_split_module_classes=["LlamaDecoderLayer"],
)
tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(checkpoint_location)


# Use model

print(tokenizer.decode(model.generate(**({ k: torch.unsqueeze(torch.tensor(v), 0) for k,v in tokenizer("Hi there, how are you doing?").items()}), max_new_tokens = 20).squeeze()))

# Prompt

base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"
