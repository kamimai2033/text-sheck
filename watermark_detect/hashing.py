import hashlib

DEFAULT_SEED = 15485863
DEFAULT_GAMMA = 0.5


def context_hash(prev_tokens, seed=DEFAULT_SEED):
    payload = str(seed) + "/" + ",".join(str(t) for t in prev_tokens)
    return hashlib.sha256(payload.encode("utf-8")).digest()


def is_green(token_id, prev_tokens, seed=DEFAULT_SEED, gamma=DEFAULT_GAMMA):
    h = hashlib.sha256(context_hash(prev_tokens, seed) + str(token_id).encode("utf-8")).digest()
    value = int.from_bytes(h[:8], "big")
    return (value / float(2**64)) < gamma
