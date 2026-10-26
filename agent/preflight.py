"""Checks run before a stage's first model call: a model-backed tool the stage cannot do without must be loadable,
or the stage stops at once instead of spending its budget on workarounds the verifiers will reject."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Runs in the tool's interpreter. Prints "OK <device>" or "FAIL <reason>" as its last line.
SAM3_PROBE = r"""
import sys
try:
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
    from sam3.model_builder import build_sam3_image_model  # noqa: F401
    from huggingface_hub import try_to_load_from_cache
except Exception as e:
    print(f"FAIL import: {type(e).__name__}: {e}"); sys.exit(0)
ckpt = try_to_load_from_cache("facebook/sam3", "sam3.pt")
if not (isinstance(ckpt, str) and ckpt):
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        get_hf_file_metadata(hf_hub_url("facebook/sam3", "sam3.pt"))
    except Exception as e:
        print(f"FAIL checkpoint: facebook/sam3 is not cached and cannot be downloaded: {type(e).__name__}: {str(e).splitlines()[-1][:300]}")
        sys.exit(0)
print("OK " + ("cuda" if torch.cuda.is_available() else "cpu"))
"""


def tool_python(cfg: dict, tool_name: str) -> str:
    envs = cfg.get("environments") or {}
    env = (cfg.get("tool_environments") or {}).get(tool_name)
    return str(envs[env]) if env and envs.get(env) else sys.executable


def check_sam3(cfg: dict, timeout: float = 300.0) -> tuple[bool, str]:
    """(ready, message) for the interpreter segment_with_text_prompts / detect_objects are routed to."""
    python = tool_python(cfg, "segment_with_text_prompts")
    where = f"{python} (environments.{(cfg.get('tool_environments') or {}).get('segment_with_text_prompts') or '<this interpreter>'})"
    if not Path(python).exists():
        return False, f"SAM 3 interpreter does not exist: {where}"
    env = dict(os.environ)
    hf = (cfg.get("models") or {}).get("hf_home")
    if hf:
        env.setdefault("HF_HOME", str(hf))
    try:
        r = subprocess.run([python, "-c", SAM3_PROBE], capture_output=True, text=True, timeout=timeout, cwd=str(REPO), env=env)
    except subprocess.TimeoutExpired:
        return False, f"SAM 3 check timed out after {timeout:.0f} s in {where}"
    last = (r.stdout.strip().splitlines() or [r.stderr.strip()[-300:] or f"exit code {r.returncode}"])[-1]
    if last.startswith("OK"):
        device = last.split()[-1]
        if device != "cuda":
            return True, f"SAM 3 ready in {where}, but on CPU (no CUDA device): segmentation will be slow"
        return True, f"SAM 3 ready in {where} on cuda"
    return False, f"SAM 3 is not usable in {where}: {last.removeprefix('FAIL ')}"
