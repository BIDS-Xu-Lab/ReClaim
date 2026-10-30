#!/usr/bin/env python3
"""
LightGBM bag-of-codes baseline for disease prediction.

Usage:
  # Train on a training split and evaluate on a held-out case/control cohort
  python run_lightgbm_baseline.py --phase all --train_data_dir ... \
      --case_control_ids_file ... --val_data_file ... --output_dir MODEL_DIR ...

  # Run phases separately
  python run_lightgbm_baseline.py --phase prepare ...
  python run_lightgbm_baseline.py --phase features ...
  python run_lightgbm_baseline.py --phase fit ...

  # Score another dataset (e.g. an external EHR cohort) with the trained models
  python run_lightgbm_baseline.py --phase eval --inference_only --output_dir MODEL_DIR \
      --inference_case_control_ids_file ... --inference_val_data_file ... \
      --inference_output_dir ...
"""

from lightgbm_baseline.run import main

if __name__ == "__main__":
    main()
