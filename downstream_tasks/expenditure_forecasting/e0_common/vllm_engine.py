"""ONE vLLM engine configuration for every vLLM consumer (MC anchor generation and the
vLLM readout), so inference-cost comparisons differ only in SamplingParams — never in
engine settings.  Also the fail-fast join for data-parallel worker pools."""
import time, logging

# The MC anchor's historical settings (generate_vllm.py); the readout inherits them.
ENGINE_DEFAULTS = dict(gpu_memory_utilization=0.8, max_model_len=4096, dtype="auto", seed=66,
                       trust_remote_code=True, enable_prefix_caching=True)


def build_llm(model_path, **overrides):
    """Returns (LLM, config-dict-actually-used).  Log the dict: it is the comparability record."""
    from vllm import LLM
    import vllm
    cfg = {**ENGINE_DEFAULTS, **overrides}
    logging.info("  vLLM %s engine: %s", vllm.__version__, {k: v for k, v in cfg.items()})
    return LLM(model=model_path, **cfg), cfg


def join_fail_fast(procs, poll_s=15.0, max_wall_s=172_800.0, grace_s=60.0) -> int:
    """The moment any worker exits non-zero, terminate the live ones (a sequential join once
    let 7 ranks run 5 h after rank 0 died at minute 6).  Returns 0 or the first failure code."""
    deadline = time.time() + max_wall_s
    while True:
        failed = [p for p in procs if p.exitcode not in (None, 0)]
        alive = [p for p in procs if p.is_alive()]
        if failed:
            code = failed[0].exitcode
            logging.error("DP rank process %d exited with code %d -> terminating the other %d live rank(s) (fail-fast)",
                          failed[0].pid, code, len(alive))
            for p in alive: p.terminate()
            for p in alive:
                p.join(timeout=grace_s)
                if p.is_alive(): p.kill()
            return code
        if not alive:
            return 0
        if time.time() > deadline:
            logging.error("48h timeout: killing %d DP rank process(es)", len(alive))
            for p in alive: p.kill()
            return 1
        time.sleep(poll_s)
