#%%
# Download model to /data/public_models cache

import os
from huggingface_hub import hf_hub_download, snapshot_download

def download_model(model_repo, cache_dir):
    repo_path = os.path.join(cache_dir, model_repo)
    os.makedirs(repo_path, exist_ok=True)

    snapshot_download(
        repo_id=model_repo,
        local_dir=repo_path,
        local_dir_use_symlinks=False,
        # ignore_patterns=["*.msgpack"],  # Exclude unnecessary files
        # allow_patterns=["*.json", "*.py", "*.bin", "*.txt", "*.h5", "*.joblib"],
        local_files_only=False,
    )

    size_bytes = sum(f.stat().st_size for f in os.scandir(repo_path) if f.is_file())
    size_mb = size_bytes / (1024 * 1024)

    return size_mb


def main():
    cache_dir = "/data/public_models"
    total_size = 0
    model_repos = [
        # "bigscience/bloom",
        "bigscience/bloom-1b1",
        "bigscience/bloom-1b7",
        "bigscience/bloom-3b",
        "bigscience/bloom-560m",
        "bigscience/bloom-7b1",
        "facebook/galactica-1.3b",
        # "facebook/galactica-120b",
        "facebook/galactica-125m",
        # "facebook/galactica-30b",
        "facebook/galactica-6.7b",
        "google/recurrentgemma-2b",
        "xai-org/grok-1",
        "openai-community/openai-gpt",
        "openai-community/gpt2",
        "openai-community/gpt2-medium",
        "openai-community/gpt2-large",
        "openai-community/gpt2-xl",
        "tiiuae/falcon-7b",
        "tiiuae/falcon-40b",
        "tiiuae/falcon-rw-1b",
        "tiiuae/falcon-rw-7b",
        # "tiiuae/falcon-180B",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mixtral-8x22B-v0.1",
        "Salesforce/ctrl",
        "mosaicml/mpt-7b",
        "mosaicml/mpt-7b-8k",
        # "mosaicml/mpt-30b",
        # "CohereForAI/c4ai-command-r-plus",
        # "CohereForAI/c4ai-command-r-plus-4bit",
        # "CohereForAI/c4ai-command-r-v01",
        # "CohereForAI/c4ai-command-r-v01-4bit",
        "stabilityai/stablelm-2-12b",
        "stabilityai/stablelm-2-1_6b",
        "stabilityai/stablelm-zephyr-3b",
        "stabilityai/stablelm-3b-4e1t",
        "stabilityai/stablelm-2-zephyr-1_6b",
        "meta-llama/Meta-Llama-3-8B",
        # "meta-llama/Meta-Llama-3-70B",
        # "LumiOpen/Viking-33B",
        "LumiOpen/Viking-13B",
        "LumiOpen/Viking-7B",
        # "LumiOpen/Poro-34B",
        "databricks/dbrx-base",
        # "xverse/XVERSE-65B-2",
        # "microsoft/Orca-2-13b",
        "microsoft/Orca-2-7b",
        "nvidia/nemotron-3-8b-base-4k",
        # "deepseek-ai/deepseek-llm-67b-base",
        "deepseek-ai/deepseek-llm-7b-base",
        "Nanbeige/Nanbeige2-8B-Chat",
        # "Nanbeige/Nanbeige-16B-Chat",
        # "Qwen/Qwen1.5-72B",
        "Qwen/Qwen1.5-14B",
        "Qwen/Qwen1.5-7B",
        "Qwen/Qwen1.5-4B",
        "Qwen/Qwen1.5-1.8B",
        "Qwen/Qwen1.5-0.5B",
        "allenai/OLMo-1.7-7B-hf",
        "allenai/OLMo-7B-hf",
        "allenai/OLMo-1B-hf",
        "allenai/OLMo-7B-SFT",
        "state-spaces/mamba-2.8b-hf",
        "deeplang-ai/LingoWhale-8B",
        # "Skywork/Skywork-13B-base",
        "adept/fuyu-8b",
        "BAAI/Aquila2-7B",
        # "BAAI/Aquila2-34B",
        # "CofeAI/FLM-101B",
        "baichuan-inc/Baichuan2-7B-Base",
        # "baichuan-inc/Baichuan-13B-Base"
    ]
    
    for model_repo in model_repos:
        print(f"Downloading model: {model_repo}")
        size_mb = download_model(model_repo, cache_dir)
        print(f"Size of {model_repo}: {size_mb:.2f} MB")
        total_size += size_mb
        
        if total_size >= 2 * 1024 * 1024:  # 2TB in MB
            print("Total size limit of 2TB reached. Stopping download.")
            break
    
    print(f"Total size of downloaded models: {total_size:.2f} MB")

if __name__ == "__main__":
    main()
# %%
