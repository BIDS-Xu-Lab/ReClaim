import torch
from transformers import Qwen3Config, GPT2Config, LlamaConfig

def create_model_config(model_type, tokenizer, max_seq_len):
    vocab_size = len(tokenizer.vocab)
    print(f'Using model: {model_type}')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU detected: {gpu_name}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024**3) # Convert to GB
            allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)
            cached_mem = torch.cuda.memory_reserved(i) / (1024**3)
            available_mem = total_mem - cached_mem
            print(f"GPU {i} ({props.name}): Total = {total_mem:.1f}GB, \
                                            Allocated = {allocated_mem:.1f}GB, \
                                            Cached={cached_mem:.1f}GB, \
                                            Available={available_mem:.1f}GB")

        if model_type == "qwen3-S":
            return Qwen3Config(
                vocab_size = vocab_size,
                hidden_size=1024,
                num_hidden_layers=8,
                num_attention_heads=16,
                num_key_value_heads=8,
                intermediate_size=2048,
                max_position_embeddings = max_seq_len,
                pad_token_id = tokenizer.pad_token_id,
                bos_token_id = tokenizer.bos_token_id,
                eos_token_id = tokenizer.eos_token_id,
            )

        elif model_type == "qwen3-M":
            return Qwen3Config(
                vocab_size = vocab_size,
                hidden_size=2048,
                num_hidden_layers=16,
                num_attention_heads=24,
                num_key_value_heads=12,
                intermediate_size=4096,
                max_position_embeddings = max_seq_len,
                pad_token_id = tokenizer.pad_token_id,
                bos_token_id = tokenizer.bos_token_id,
                eos_token_id = tokenizer.eos_token_id,
            )
        
        elif model_type == "qwen3-L":
            return Qwen3Config(
                vocab_size = vocab_size,
                hidden_size=2048,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=16,
                intermediate_size=4096,
                max_position_embeddings = max_seq_len,
                pad_token_id = tokenizer.pad_token_id,
                bos_token_id = tokenizer.bos_token_id,
                eos_token_id = tokenizer.eos_token_id,
            )

        else:
            raise ValueError(f"Unknown model type {model_type}")

    else:
        raise ValueError(f"No CUDA GPUs available")