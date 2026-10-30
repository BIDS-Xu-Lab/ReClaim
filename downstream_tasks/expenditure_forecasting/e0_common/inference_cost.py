"""Inference-cost accounting shared by every readout / generation path (engine-neutral).

Two numbers per (model, section), written to <section>/model_gen/<model>/timing.json
and tabulated by e4_eval/summarize_inference_cost.py:
  gpu_s_per_patient = wall_s * n_gpus / patients
  gpu_h             = wall_s * n_gpus / 3600            (per section; the eval total is the sum)
Wall time is the section's end-to-end time inside the job INCLUDING engine/model start-up
(one-time, ~1 min — <1% at cohort scale) so the HF, vLLM and MC paths are accounted
identically.  Token counts are engine-free and let the algorithmic saving be reported
independently of engine efficiency.
"""
import os, json, time, logging, datetime


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


class SectionTimer:
    def __enter__(self):
        self.t0 = time.time(); return self
    def __exit__(self, *a):
        self.wall_s = time.time() - self.t0


def timing_record(*, model, section, engine, patients, wall_s, n_gpus, tokens_generated_per_patient=None,
                  prompt_tokens=None, **extra):
    gpu_s = wall_s * n_gpus
    rec = dict(model=model, section=section, engine=engine, gpu=gpu_name(), n_gpus=int(n_gpus),
               patients=int(patients), wall_s=round(wall_s, 1), gpu_s=round(gpu_s, 1),
               gpu_s_per_patient=round(gpu_s / max(patients, 1), 5), gpu_h=round(gpu_s / 3600, 3),
               tokens_generated_per_patient=tokens_generated_per_patient, prompt_tokens=prompt_tokens,
               recorded=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), **extra)
    logging.info("  TIMING [%s] %s: %d patients, %d GPU(s), wall %.0fs -> %.4f GPU-s/patient, %.3f GPU-h%s",
                 section, engine, patients, n_gpus, wall_s, rec["gpu_s_per_patient"], rec["gpu_h"],
                 f", {tokens_generated_per_patient:.1f} gen tok/patient" if tokens_generated_per_patient is not None else "")
    return rec


def log_evaluation_total(model, recs):
    """One summary line at the end of a job, from the section records it wrote (the same
    numbers e4 tabulates as the evaluation_total row)."""
    if not recs:
        return
    gpu_s = sum(r["gpu_s"] for r in recs); pats = sum(r["patients"] for r in recs)
    logging.info("  TIMING [evaluation_total] %s: %s on %s x%d: %d patients, wall %.0fs, %.4f GPU-s/patient, %.3f GPU-h  (sections: %s)",
                 model, recs[0]["engine"], recs[0]["gpu"], recs[0]["n_gpus"], pats, sum(r["wall_s"] for r in recs),
                 gpu_s / max(pats, 1), gpu_s / 3600, ", ".join(r["section"] for r in recs))


def write_timing(model_gen_dir, rec, filename="timing.json"):
    """<filename> holds ONE record per section for this model dir (merge-on-save).
    Post-train readouts write timing.json; the MC generator passes timing_<N>.json so the
    cost of every trajectory count survives side by side, like cost_predictions_<N>.csv and
    generated_data_<N>.parquet."""
    os.makedirs(model_gen_dir, exist_ok=True)
    path = os.path.join(model_gen_dir, filename)
    recs = []
    if os.path.exists(path):
        try:
            recs = [r for r in json.load(open(path)) if r.get("section") != rec["section"]]
        except Exception:
            recs = []
    recs.append(rec)
    json.dump(recs, open(path, "w"), indent=1)
    return path


def local_bin_artifact(ref_path, data_dir):
    """A checkpoint's bin_artifact_ref.json carries the path of the cluster that trained it.
    If it does not exist here, use the same artifact under this cluster's DATA_DIR
    (data/<pairs dir>/<file>): identical content, since the pairs are content-keyed."""
    if os.path.exists(ref_path):
        return ref_path
    cand = os.path.join(data_dir, os.path.basename(os.path.dirname(ref_path)), os.path.basename(ref_path))
    if os.path.exists(cand):
        logging.warning("  Bin artifact ref points to another cluster (%s); using local %s", ref_path, cand)
        return cand
    raise FileNotFoundError(f"bin artifact not found at {ref_path} nor {cand}")
