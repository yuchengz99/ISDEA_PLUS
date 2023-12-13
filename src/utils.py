import hashlib
from datetime import datetime

def create_hash(params_string: str) -> str:
    R"""
    Create hash from command line argument string. This is mainly for logging purposes.
    """
    hasher = hashlib.md5()
    hasher.update(params_string.encode('utf-8'))
    raw_hash =  hasher.hexdigest()
    hash_str = "{}".format(raw_hash)[:8]
    return hash_str

def wandb_run_name(
    run_hash: str,
    stage_name: str
) -> str:
    R"""
    Create a run name for Weights & Biases.
    """
    return f"{run_hash} ({stage_name}) @ {datetime.now().strftime('%m%d%Y|%H:%M:%S')}"