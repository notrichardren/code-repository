# llama-2-70b-hf-inference

How to do llama-70b HuggingFace inference, parallelized across multiple GPUs

Need to set ```accelerate config```. Then, run ```inference.py``` to download a model into a given directory and run load_checkpoint_and_dispatch in the accelerate library.

Want to expand to have a codebase for:
- OpenAI inference class
- Llama 70b hf inference (accelerate, or just normal)
- Llama inference
- Mistral inference
- All other inference with chat template that works
- Generic evaluation, maybe.
- Getting folder file structure
- SSH commands
