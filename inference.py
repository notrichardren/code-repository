#%%
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download, snapshot_download
import torch

model_name = f"meta-llama/Llama-2-70b-chat-hf"

weights_dir = f"{os.getcwd()}/llama-weights-70b"
if not os.path.exists(weights_dir):
    os.system(f"mkdir {weights_dir}")

checkpoint_location = snapshot_download(model_name, local_dir=weights_dir, ignore_patterns=["*.safetensors", "model.safetensors.index.json"])
checkpoint_location = weights_dir

with init_empty_weights():
    model = LlamaForCausalLM.from_pretrained(checkpoint_location)

model = load_checkpoint_and_dispatch(
    model,
    checkpoint_location,
    device_map="auto",
    offload_folder=weights_dir,
    dtype=torch.float16,
    no_split_module_classes=["LlamaDecoderLayer"],
)
tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained(checkpoint_location)
