# llama-2-70b-hf-inference

How to do llama-70b HuggingFace inference, parallelized across multiple GPUs

Need to set ```accelerate config```. Then, run ```inference.py``` to download a model into a given directory and run load_checkpoint_and_dispatch in the accelerate library.
