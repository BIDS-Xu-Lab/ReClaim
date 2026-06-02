#!/usr/bin/env python3
"""
Extract patient embeddings from ReClaim foundation model.

This script processes clinical token sequences and generates embeddings using
a pre-trained ReClaim model with last-token pooling (appropriate for causal LMs).

Embeddings are L2 normalized to unit vectors, making dot product equivalent to
cosine similarity. This is optimal for patient similarity search and propensity
score matching augmentation.

Truncation Strategy:
- Default (left truncation): Keeps last tokens (recent events)
  [truncated from left] ... <VISIT-XXX> <DIAG-XXX> <PROC-XXX>
- Customized truncation: Preserves both start (demographics) AND end (recent events)
"""

import pandas as pd
import numpy as np
import os
import argparse
import logging
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from utils import setup_logging, time_execution

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract patient embeddings from ReClaim foundation model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str,
                       default='v6-s',
                       help='Model')
    parser.add_argument('--model_path', type=str,
                       default=os.environ.get("RECLAIM_MODEL_PATH", "/path/to/reclaim_model_checkpoint"),
                       help='Directory containing models')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--max_length', type=int, default=4096,
                       help='Maximum sequence length')
    parser.add_argument('--customized_truncate', action='store_true',
                       help='Use customized truncation preserving both start (demographics) '
                            'and end (recent events). Default uses left truncation '
                            '(keeps last tokens/recent events).')
    parser.add_argument('--keep_start', type=int, default=4,
                       help='Number of tokens to preserve from start when using '
                            'customized truncation. Default 4 for: '
                            '<sos> <SEX-X> <DOBYR-XXXX> <AGE-XX>')
    parser.add_argument('--random_seed', type=int, default=66,
                       help='Random seed for reproducibility')
    return parser.parse_args()


def customized_truncate(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_length: int = 4096,
    keep_start: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Truncate sequence preserving both start (demographics) and end (recent events).
    
    For clinical sequences where:
    - Start contains critical demographics: <sos> <SEX-1> <DOBYR-1968> <AGE-40> ...
    - End contains recent events with high predictive value
    
    Parameters:
    -----------
    input_ids : torch.Tensor
        Token IDs of shape [batch, seq_len]
    attention_mask : torch.Tensor
        Attention mask of shape [batch, seq_len]
    max_length : int
        Maximum sequence length to truncate to
    keep_start : int
        Number of tokens to preserve from the start (demographics)
        
    Returns:
    --------
    tuple of truncated (input_ids, attention_mask)
    """
    seq_len = input_ids.shape[1]
    
    if seq_len <= max_length:
        return input_ids, attention_mask
    
    keep_end = max_length - keep_start
    
    # Concatenate start and end portions
    input_ids_truncated = torch.cat([
        input_ids[:, :keep_start],
        input_ids[:, -keep_end:]
    ], dim=1)
    
    attention_mask_truncated = torch.cat([
        attention_mask[:, :keep_start],
        attention_mask[:, -keep_end:]
    ], dim=1)
    
    return input_ids_truncated, attention_mask_truncated


def normalize_embeddings(
    embeddings: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    L2 normalize embeddings to unit vectors.
    
    After normalization, dot product equals cosine similarity,
    which is useful for similarity search and clustering.
    
    Parameters:
    -----------
    embeddings : np.ndarray
        Embeddings of shape [n_samples, hidden_dim]
    eps : float
        Small constant to avoid division by zero
        
    Returns:
    --------
    np.ndarray of shape [n_samples, hidden_dim] with unit norm
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)  # Avoid division by zero
    return embeddings / norms


def get_last_token_embeddings(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Extract the last non-padding token's hidden state for each sequence.
    
    For causal models, only the last token has seen the entire sequence,
    making it the appropriate choice for sequence-level representation.
    
    Parameters:
    -----------
    hidden_states : torch.Tensor
        Hidden states of shape [batch, seq_len, hidden_dim]
    attention_mask : torch.Tensor
        Attention mask of shape [batch, seq_len], 1 for real tokens, 0 for padding
        
    Returns:
    --------
    torch.Tensor of shape [batch, hidden_dim]
    """
    batch_size = hidden_states.shape[0]
    
    # Find index of last real token for each sequence
    # Sum of attention_mask gives sequence length, subtract 1 for 0-indexing
    last_nonpad_token_indices = attention_mask.sum(dim=1) - 1  # Shape: [batch]
    
    # Gather last token embedding for each sequence
    last_token_embeddings = hidden_states[
        torch.arange(batch_size, device=hidden_states.device),
        last_nonpad_token_indices.long(),  # guaranteed to integer type
        :
    ]
    
    return last_token_embeddings


@time_execution
def add_reclaim_embeddings_to_cohort(
    input_file: str,
    output_file: str,
    model_path: str,
    batch_size: int = 8,
    max_length: int = 4096,
    use_customized_truncate: bool = False,
    keep_start: int = 4,
) -> pd.DataFrame:
    """
    Add ReClaim embedding columns to the cohort dataframe using pre_entry_seq.
    
    Uses last-token pooling appropriate for causal/decoder models like ReClaim,
    where only the final token has seen the entire sequence context.
    
    Embeddings are L2 normalized to unit vectors, making dot product equivalent
    to cosine similarity. This is optimal for patient similarity search and
    propensity score matching augmentation.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file (cohort_with_pre_entry_seq.csv)
    output_file : str
        Path to the output CSV file with embeddings added
    model_path : str
        Path to the ReClaim model
    batch_size : int
        Batch size for inference (adjust based on GPU memory)
    max_length : int
        Maximum sequence length (ReClaim supports up to 4096)
    use_customized_truncate : bool
        If True, preserve both start (demographics) and end (recent events).
        If False (default), use left truncation (keep last tokens/recent events).
    keep_start : int
        Number of tokens to preserve from start when using customized truncation.
        Default is 4 for: <sos> <SEX-X> <DOBYR-XXXX> <AGE-XX>
        
    Returns:
    --------
    pandas.DataFrame: Dataframe with embedding columns added (normalized to unit vectors)
    """
    logging.info("Loading cohort data from %s", input_file)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using device: %s", device)
    
    if device.type == 'cuda':
        logging.info("GPU: %s", torch.cuda.get_device_name(0))
        logging.info("GPU Memory: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    
    # Load cohort data
    cohort_df = pd.read_csv(input_file, low_memory=False)
    logging.info("Loaded cohort data: %s patients x %s columns", 
                 f"{cohort_df.shape[0]:,}", f"{cohort_df.shape[1]:,}")
    
    # Check for pre_entry_seq column
    if 'pre_entry_seq' not in cohort_df.columns:
        raise ValueError("pre_entry_seq column not found. "
                        "Ensure using the correct input file cohort_with_pre_entry_seq.csv")
    
    # Filter out rows with null pre_entry_seq
    cohort_df_filtered = cohort_df.dropna(subset=['pre_entry_seq']).reset_index(drop=True)
    null_count = len(cohort_df) - len(cohort_df_filtered)
    if null_count > 0:
        logging.warning("Filtered out %s patients with null pre_entry_seq", f"{null_count:,}")
    logging.info("Processing %s patients with valid pre_entry_seq", 
                 f"{cohort_df_filtered.shape[0]:,}")
    
    # Load tokenizer
    logging.info("Loading ReClaim tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Set padding token if not set (common for causal LMs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token: %s", tokenizer.pad_token)
    
    # Set truncation side to LEFT to keep last tokens (recent events) by default
    tokenizer.truncation_side = "left"
    
    sequences = cohort_df_filtered['pre_entry_seq'].tolist()
    
    # Load model
    logging.info("Loading ReClaim model from %s", model_path)
    
    # Determine dtype (use bfloat16 if supported, otherwise float16)
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        logging.info("Using bfloat16 precision")
    else:
        dtype = torch.float16
        logging.info("Using float16 precision")
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map='auto' if device.type == 'cuda' else None,
    )
    
    # Move to device if not using device_map
    if device.type != 'cuda':
        model = model.to(device)
    
    # Set to evaluation mode (critical for deterministic behavior)
    model.eval()
    logging.info("Model loaded and set to eval mode")
    
    # Get model config for embedding dimension
    config = model.config
    hidden_size = config.hidden_size
    logging.info("Model hidden size: %s", hidden_size)
    logging.info("Model max position embeddings: %s", 
                 getattr(config, 'max_position_embeddings', 'N/A'))
    
    # Log settings
    logging.info("Settings: batch_size=%s, max_length=%s, use_customized_truncate=%s, keep_start=%s", 
                 batch_size, max_length, use_customized_truncate, keep_start)
    
    # Process in batches
    logging.info("Generating embeddings...")
    all_embeddings = []
    
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sequences))
        batch_sequences = sequences[start_idx:end_idx]
        
        # Tokenize batch
        if use_customized_truncate:
            # No truncation during tokenization - we'll handle it manually
            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=False,
                max_length=None,
            )
        else:
            # Use tokenizer's left truncation (keeps last tokens/recent events)
            inputs = tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
        
        # Remove token_type_ids if present (not used by ReClaim)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Apply customized truncation if needed (preserves both start and end)
        if use_customized_truncate and input_ids.shape[1] > max_length:
            input_ids, attention_mask = customized_truncate(
                input_ids, 
                attention_mask, 
                max_length=max_length,
                keep_start=keep_start,
            )
        
        # Move to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            
            # Get last layer hidden states
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
            
            # Get sequence embeddings (last token)
            batch_embeddings = get_last_token_embeddings(hidden_states, attention_mask)
            
            # Move to CPU and convert to numpy
            batch_embeddings = batch_embeddings.float().cpu().numpy()
            all_embeddings.append(batch_embeddings)
        
        # Clear GPU cache periodically
        if device.type == 'cuda' and batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    embeddings_array = np.vstack(all_embeddings)
    logging.info("Generated embeddings with shape: %s", embeddings_array.shape)
    
    # L2 normalize embeddings to unit vectors
    # This makes dot product = cosine similarity, optimal for patient similarity search
    embeddings_array = normalize_embeddings(embeddings_array)
    logging.info("Normalized embeddings to unit vectors (L2 norm = 1)")
    
    # Verify normalization
    norms = np.linalg.norm(embeddings_array, axis=1)
    logging.info("Embedding norms - min: %.6f, max: %.6f, mean: %.6f", 
                 norms.min(), norms.max(), norms.mean())
    
    # Create embedding dataframe
    embedding_df = pd.DataFrame(
        embeddings_array,
        columns=[f'rep_{i}' for i in range(embeddings_array.shape[1])]
    )
    
    # Concatenate with original data
    cohort_df_with_embedding = pd.concat([
        cohort_df_filtered.reset_index(drop=True),
        embedding_df.reset_index(drop=True),
    ], axis=1)
    
    logging.info("Final dataframe: %s patients x %s columns",
                f"{cohort_df_with_embedding.shape[0]:,}", 
                f"{cohort_df_with_embedding.shape[1]:,}")
    
    # Save results
    logging.info("Saving to %s", output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    cohort_df_with_embedding.to_csv(output_file, index=False)
    logging.info("Save complete!")
    
    return cohort_df_with_embedding


@time_execution
def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging using utils.py
    log_file = setup_logging()
    
    logging.info("Patient Embedding Extraction Script (ReClaim)")
    logging.info("Log file: %s", log_file)
    
    # Set random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    
    # Construct file paths
    working_dir = os.getcwd()
    
    # Extract version and size from model name (e.g., "v6-s" -> version=6, size="s")
    model_parts = args.model.split('-')
    version = int(model_parts[0][1:])  # Extract version number (e.g., "v6" -> 6)
    model_size = model_parts[1] if len(model_parts) > 1 else None  # Extract size (e.g., "s", "m", "l", "no")
    
    # Common file paths
    final_cohort_dir = os.path.join(working_dir, 'intermediate', f'v{version}', 'final_cohort')
    input_file = os.path.join(final_cohort_dir, "cohort_with_pre_entry_seq.csv")
    output_file = os.path.join(final_cohort_dir, f"cohort_with_{args.model}_embeddings.csv")
    
    logging.info("Working directory: %s", working_dir)
    logging.info("Model: %s (version=%d, size=%s)", args.model, version, model_size)
    logging.info("Input file: %s", input_file)
    logging.info("Output file: %s", output_file)
    
    # Check input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Handle 'no' model size - save data without embeddings
    if model_size == 'no':
        logging.info("No embeddings mode - copying data without embeddings")
        
        cohort_df = pd.read_csv(input_file)
        logging.info("Loaded %s patients", f"{cohort_df.shape[0]:,}")
        
        # Drop pre_entry_seq column
        if 'pre_entry_seq' in cohort_df.columns:
            cohort_df = cohort_df.drop(columns=['pre_entry_seq'])
            logging.info("Dropped pre_entry_seq column")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        cohort_df.to_csv(output_file, index=False)
        logging.info("Saved to %s", output_file)
    
    else:
        # Construct model path from model_dir and model name
        model_path = args.model_path
        
        # Check model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logging.info("Model path: %s", model_path)
        logging.info("Batch size: %s", args.batch_size)
        logging.info("Max length: %s", args.max_length)
        if args.customized_truncate:
            logging.info("Truncation: customized (keep first %d + last %d tokens)", 
                        args.keep_start, args.max_length - args.keep_start)
        else:
            logging.info("Truncation: left (keep last %d tokens/recent events)", 
                        args.max_length)
        
        add_reclaim_embeddings_to_cohort(
            input_file=input_file,
            output_file=output_file,
            model_path=model_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_customized_truncate=args.customized_truncate,
            keep_start=args.keep_start,
        )
    
    logging.info("Embedding extraction complete!")


if __name__ == "__main__":
    main()
