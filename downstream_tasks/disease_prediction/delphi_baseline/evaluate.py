"""
Delphi Model Evaluation on a Predefined Case/Control Cohort

Scores a trained Delphi checkpoint on the cohort prepared by
transform_predefined_cohort.py and writes per-disease AUCs.

Pipeline: Load model + data → prefix inference → stratified AUC → CSV output
"""

import os
import argparse
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import logging
import json
from datetime import datetime


# === Logging Setup ===
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filepath = os.path.join(log_dir, f"evaluate_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filepath)
    ]
)
logger = logging.getLogger("DelphiEvaluation")
logger.info(f"Log file saved to: {log_filepath}")

# Ensure local modules are importable
sys.path.append(os.getcwd())

try:
    from model import DelphiConfig, Delphi
    from utils import get_p2i
    from evaluate_auc import evaluate_auc_pipeline_predefined
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    logger.error("Ensure model.py, utils.py, and evaluate_auc.py are in the working directory.")
    sys.exit(1)


def summarize_predefined_record(record, limit=3):
    return {
        "disease_token": int(record["disease_token"]),
        "disease_code": record.get("disease_code"),
        "sex": record["sex"],
        "age": int(record["age"]),
        "n_cases": len(record.get("case_patient_indices", [])),
        "n_controls": len(record.get("control_patient_indices", [])),
        "case_examples": [
            {"patient_idx": int(idx), "seq_point": int(seq_pt)}
            for idx, seq_pt in zip(record.get("case_patient_indices", [])[:limit], record.get("case_seq_points", [])[:limit])
        ],
        "control_examples": [
            {"patient_idx": int(idx), "seq_point": int(seq_pt)}
            for idx, seq_pt in zip(record.get("control_patient_indices", [])[:limit], record.get("control_seq_points", [])[:limit])
        ],
    }


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_arguments():
    """Parse command line arguments for the Delphi evaluation script."""
    parser = argparse.ArgumentParser(description="Delphi Model Evaluation Pipeline")

    # --- Model ---
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory containing the model checkpoint (ckpt.pt)')

    # --- Data ---
    parser.add_argument('--val_path', type=str, required=True,
                        help='Path to the Delphi-format .bin written by preprocess.py')
    parser.add_argument('--predefined_cohort', type=str, required=True,
                        help='Path to prepared predefined cohort JSON (from transform_predefined_cohort.py)')

    # --- Evaluation parameters ---
    parser.add_argument('--offset', type=float, default=365.25,
                        help='Prediction offset in days, used only to name outputs (default: 365.25)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device: 'cuda', 'cpu', or 'mps' (default: auto-detect)")
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for predefined-prefix inference (default: 64)')

    # --- Output ---
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save result CSVs (default: .)')

    # --- Advanced ---
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed (default: 1337)')
    parser.add_argument('--labels_file', type=str, default='delphi_labels_chapters_colours_icd.csv',
                        help='Path to disease labels CSV file')
    parser.add_argument('--debug_examples', type=int, default=5,
                        help='Number of representative predefined-cohort examples to log (default: 5)')

    return parser.parse_args()


# =============================================================================
# Main Pipeline
# =============================================================================

def icd_code_from_name(name):
    """'A02 Other salmonella infections' -> 'A02'."""
    return str(name).split(" ", 1)[0]


def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = Path(args.out_dir).name
    data_name = Path(args.val_path).parent.name
    detail_csv_path = os.path.join(
        args.output_dir, f"auc_{model_name}_{data_name}_offset_{int(args.offset)}_predefined.csv"
    )

    logger.info("--- Configuration Summary ---")
    logger.info(f"Model Directory   : {args.out_dir}")
    logger.info(f"Data Path         : {args.val_path}")
    logger.info(f"Predefined Cohort : {args.predefined_cohort}")
    logger.info(f"Output Directory  : {args.output_dir}")
    logger.info(f"Device            : {args.device}")
    logger.info(f"Debug Examples    : {args.debug_examples}")

    # --- Set seeds ---
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed(args.seed)

    # --- Load labels ---
    labels_path = args.labels_file
    if not os.path.exists(labels_path):
        labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.labels_file)
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {args.labels_file}")
    delphi_labels = pd.read_csv(labels_path)

    # --- Load model ---
    logger.info(f"Loading model from {args.out_dir}...")
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=args.device)
    conf = DelphiConfig(**checkpoint['model_args'])
    model = Delphi(conf)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.to(args.device)

    # --- Load data ---
    logger.info(f"Loading data from {args.val_path}...")
    if not os.path.exists(args.val_path):
        raise FileNotFoundError(f"Data file not found: {args.val_path}")
    val = np.fromfile(args.val_path, dtype=np.uint32)
    if val.size == 0:
        raise ValueError(
            f"Evaluation binary is empty: {args.val_path}. This usually means preprocessing filtered out every patient."
        )
    if val.size % 3 != 0:
        raise ValueError(f"Unexpected evaluation binary size in {args.val_path}: {val.size} uint32 values")
    val = val.reshape(-1, 3)
    val_p2i = get_p2i(val)
    logger.info(f"Dataset: {len(val_p2i)} patients")

    # --- Run evaluation ---
    logger.info(f"Loading predefined cohort from {args.predefined_cohort}...")
    if not os.path.exists(args.predefined_cohort):
        raise FileNotFoundError(f"Predefined cohort file not found: {args.predefined_cohort}")
    with open(args.predefined_cohort, 'r') as f:
        predefined_cohort = json.load(f)
    logger.info(f"Loaded predefined cohort with {len(predefined_cohort)} strata")
    if args.debug_examples and predefined_cohort:
        sample_records = [
            summarize_predefined_record(record)
            for record in predefined_cohort[:args.debug_examples]
        ]
        logger.info("Sample predefined strata: %s", json.dumps(sample_records, indent=2))

    logger.info("Running AUC evaluation with predefined cohort prefixes...")
    df_auc_unpooled_merged, df_auc_merged = evaluate_auc_pipeline_predefined(
        model,
        val=val,
        val_p2i=val_p2i,
        predefined_cohort=predefined_cohort,
        delphi_labels=delphi_labels[13:],
        offset=args.offset,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        debug_examples=args.debug_examples,
    )

    # --- Save results ---
    # Detailed per-disease table (Delphi token, labels, chapter, ...).
    df_auc_merged.to_csv(detail_csv_path)
    logger.info(f"Saved detailed results to {detail_csv_path}")

    # Same layout as test_case_controls.py, keyed by ICD-10 code so it can be
    # joined with other methods (visualization/disease_prediction/ reads `icd_code`).
    pooled = df_auc_merged.assign(icd_code=df_auc_merged["name"].map(icd_code_from_name))
    pooled = pooled[["icd_code", "auc", "n_samples", "n_diseased", "n_healthy"]].sort_values("icd_code")
    pooled.to_csv(os.path.join(args.output_dir, "auc_results.csv"), index=False)
    unpooled = df_auc_unpooled_merged.assign(icd_code=df_auc_unpooled_merged["name"].map(icd_code_from_name))
    unpooled = unpooled[["icd_code", "sex", "age", "auc_delong", "n_diseased", "n_healthy"]]
    unpooled.to_csv(os.path.join(args.output_dir, "auc_results_unpooled.csv"), index=False)
    logger.info(f"Saved auc_results.csv and auc_results_unpooled.csv to {args.output_dir}")
    logger.info(f"Diseases: {len(pooled)}. Mean AUC: {pooled['auc'].mean():.4f}")


if __name__ == "__main__":
    main()
