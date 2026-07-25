"""
Turns the from-scratch NumPy transformer into an actual NLP model: a
character-level language model trained on real text, with a real
tokenizer (character vocabulary built from the corpus) and text
generation (temperature + top-k sampling) at the end.

This sandbox has no network access, so there's no downloading a corpus
or a pretrained tokenizer. The training text below is an original short
story I wrote for this demo (no copyright concerns), long enough to give
a tiny char-level model real structure to learn: word boundaries, common
letter sequences, punctuation patterns, etc.

Reuses the same hand-derived-backprop building blocks (Linear, LayerNorm,
GELU, causal multi-head attention, Adam) as numpy_transformer_demo.py --
only the tokenizer, dataset, and generation routine are new.

Run with:
    python3 nlp_char_transformer.py
"""

import numpy as np
from scipy.special import erf

rng = np.random.default_rng(0)


# ==========================================================================
# 1. Tokenizer: build a character-level vocabulary from the corpus
# ==========================================================================

class CharTokenizer:
    """Maps characters <-> integer ids. Character-level so the vocabulary
    is tiny (a few dozen symbols) and needs no external files."""

    def __init__(self, corpus: str):
        chars = sorted(set(corpus))
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.stoi[ch] for ch in text], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


CORPUS = """The old lighthouse stood at the edge of the cliff, its lamp turning slowly through the fog every night. Sailors who passed that stretch of coast said the light seemed to follow their boats, guiding them gently around the rocks hidden beneath the waves. The keeper who lived there was an old woman named Elena, who had tended the lamp for forty years. She woke before dawn each morning, climbed the narrow spiral stairs, and checked the oil, the glass, and the wind. In storms, she stayed awake all night, watching the horizon for ships in trouble. The villagers below trusted her more than they trusted the weather reports on the radio. They said Elena could smell a storm two days before it arrived, long before the clouds ever gathered. Every autumn, when the fishing boats returned for the last time before winter, the whole village gathered at the lighthouse to thank her with bread, wine, and songs. Elena never wanted anything in return. She said the sea had given her a home when she had nowhere else to go, and the least she could do was keep its light burning. Children would visit her in the summer, asking to see the great lamp up close. She always let them turn the crank that wound the clockwork mechanism, laughing as they marveled at how something so old could still turn so smoothly. When Elena grew too old to climb the stairs herself, the village built a small elevator inside the tower, so she could keep her post a little longer. She never left the lighthouse, not even when her children asked her to move into town where it would be warmer and safer. The light, she said, was not just a job. It was a promise she had made to every sailor who ever looked toward the shore in the dark, hoping to find their way home. Long after Elena was gone, the villagers kept the lamp burning in her memory, and every child who visited the lighthouse was told her story, so that the promise would never be forgotten."""


# ==========================================================================
# 2. Model building blocks (same math as numpy_transformer_demo.py)
# ==========================================================================

def linear_forward(x, W, b):
    out = x @ W + b
    return out, (x, W)


def linear_backward(dout, cache):
    x, W = cache
    in_dim, out_dim = W.shape
    x_flat = x.reshape(-1, in_dim)
    dout_flat = dout.reshape(-1, out_dim)
    dW = x_flat.T @ dout_flat
    db = dout_flat.sum(axis=0)
    dx = (dout_flat @ W.T).reshape(x.shape)
    return dx, dW, db


def gelu_forward(x):
    cdf = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    return x * cdf, x


def gelu_backward(dout, x):
    cdf = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    return dout * (cdf + x * pdf)


def layernorm_forward(x, gamma, beta, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    xhat = (x - mu) / std
    out = gamma * xhat + beta
    return out, (xhat, std, gamma)


def layernorm_backward(dout, cache):
    xhat, std, gamma = cache
    D = xhat.shape[-1]
    dgamma = (dout * xhat).reshape(-1, D).sum(axis=0)
    dbeta = dout.reshape(-1, D).sum(axis=0)
    dxhat = dout * gamma
    dx = (1.0 / std) * (
        dxhat
        - dxhat.mean(axis=-1, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)
    )
    return dx, dgamma, dbeta


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def mha_forward(x, Wqkv, Wo, num_heads):
    b, s, e = x.shape
    h = num_heads
    d = e // h
    qkv = x @ Wqkv
    q, k, v = np.split(qkv, 3, axis=-1)

    def to_heads(t):
        return t.reshape(b, s, h, d).transpose(0, 2, 1, 3)

    qh, kh, vh = to_heads(q), to_heads(k), to_heads(v)
    scores = np.einsum("bhid,bhjd->bhij", qh, kh) / np.sqrt(d)
    causal_mask = np.triu(np.ones((s, s), dtype=bool), k=1)
    scores = np.where(causal_mask, -1e9, scores)
    attn = softmax(scores, axis=-1)
    outh = np.einsum("bhij,bhjd->bhid", attn, vh)
    merged = outh.transpose(0, 2, 1, 3).reshape(b, s, e)
    out = merged @ Wo
    cache = (x, Wqkv, Wo, qh, kh, vh, attn, merged, h, d)
    return out, cache


def mha_backward(dout, cache):
    x, Wqkv, Wo, qh, kh, vh, attn, merged, h, d = cache
    b, s, e = x.shape
    dWo = merged.reshape(-1, e).T @ dout.reshape(-1, e)
    dmerged = dout @ Wo.T
    d_outh = dmerged.reshape(b, s, h, d).transpose(0, 2, 1, 3)
    dvh = np.einsum("bhij,bhid->bhjd", attn, d_outh)
    dattn = np.einsum("bhid,bhjd->bhij", d_outh, vh)
    sum_term = (attn * dattn).sum(axis=-1, keepdims=True)
    dscores = attn * (dattn - sum_term)
    dqh = np.einsum("bhij,bhjd->bhid", dscores, kh) / np.sqrt(d)
    dkh = np.einsum("bhij,bhid->bhjd", dscores, qh) / np.sqrt(d)

    def from_heads(t):
        return t.transpose(0, 2, 1, 3).reshape(b, s, h * d)

    dq, dk, dv = from_heads(dqh), from_heads(dkh), from_heads(dvh)
    dqkv = np.concatenate([dq, dk, dv], axis=-1)
    dWqkv = x.reshape(-1, e).T @ dqkv.reshape(-1, 3 * e)
    dx = dqkv @ Wqkv.T
    return dx, dWqkv, dWo


# ==========================================================================
# 3. Model: init / forward / backward (identical architecture to before,
#    just parameterized by the tokenizer's vocab_size)
# ==========================================================================

def init_params(vocab_size, embed_dim, num_heads, num_layers, ffn_expansion, max_seq_len):
    def xavier(fan_in, fan_out):
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return rng.uniform(-limit, limit, size=(fan_in, fan_out))

    params = {
        "token_emb": rng.normal(0, 0.02, size=(vocab_size, embed_dim)),
        "pos_emb": rng.normal(0, 0.02, size=(max_seq_len, embed_dim)),
    }
    ffn_hidden = embed_dim * ffn_expansion
    for layer in range(num_layers):
        p = f"layer{layer}."
        params[p + "ln1_gamma"] = np.ones(embed_dim)
        params[p + "ln1_beta"] = np.zeros(embed_dim)
        params[p + "Wqkv"] = xavier(embed_dim, 3 * embed_dim)
        params[p + "Wo"] = xavier(embed_dim, embed_dim)
        params[p + "ln2_gamma"] = np.ones(embed_dim)
        params[p + "ln2_beta"] = np.zeros(embed_dim)
        params[p + "W1"] = xavier(embed_dim, ffn_hidden)
        params[p + "b1"] = np.zeros(ffn_hidden)
        params[p + "W2"] = xavier(ffn_hidden, embed_dim)
        params[p + "b2"] = np.zeros(embed_dim)
    params["ln_f_gamma"] = np.ones(embed_dim)
    params["ln_f_beta"] = np.zeros(embed_dim)
    return params


def forward(params, token_ids, num_heads, num_layers):
    b, s = token_ids.shape
    caches = {"embed_input": token_ids}
    x = params["token_emb"][token_ids] + params["pos_emb"][:s][None, :, :]

    for layer in range(num_layers):
        p = f"layer{layer}."
        a, ln1_cache = layernorm_forward(x, params[p + "ln1_gamma"], params[p + "ln1_beta"])
        attn_out, mha_cache = mha_forward(a, params[p + "Wqkv"], params[p + "Wo"], num_heads)
        x1 = x + attn_out

        bnorm, ln2_cache = layernorm_forward(x1, params[p + "ln2_gamma"], params[p + "ln2_beta"])
        h1, lin1_cache = linear_forward(bnorm, params[p + "W1"], params[p + "b1"])
        h1_act, gelu_cache = gelu_forward(h1)
        ffn_out, lin2_cache = linear_forward(h1_act, params[p + "W2"], params[p + "b2"])
        x2 = x1 + ffn_out

        caches[p] = (ln1_cache, mha_cache, ln2_cache, lin1_cache, gelu_cache, lin2_cache, x, x1)
        x = x2

    x_final, lnf_cache = layernorm_forward(x, params["ln_f_gamma"], params["ln_f_beta"])
    caches["lnf_cache"] = lnf_cache
    caches["hidden_final"] = x_final
    logits = x_final @ params["token_emb"].T
    return logits, caches


def cross_entropy_loss(logits, targets):
    b, s, V = logits.shape
    probs = softmax(logits, axis=-1)
    n = b * s
    log_probs = np.log(np.clip(probs[np.arange(b)[:, None], np.arange(s)[None, :], targets], 1e-12, None))
    loss = -log_probs.mean()
    dlogits = probs.copy()
    dlogits[np.arange(b)[:, None], np.arange(s)[None, :], targets] -= 1.0
    dlogits /= n
    return loss, dlogits


def backward(params, caches, dlogits, num_heads, num_layers):
    grads = {k: np.zeros_like(v) for k, v in params.items()}
    hidden_final = caches["hidden_final"]
    b, s, V = dlogits.shape

    grads["token_emb"] += dlogits.reshape(-1, V).T @ hidden_final.reshape(-1, hidden_final.shape[-1])
    dhidden_final = dlogits @ params["token_emb"]

    dx, dgamma, dbeta = layernorm_backward(dhidden_final, caches["lnf_cache"])
    grads["ln_f_gamma"] += dgamma
    grads["ln_f_beta"] += dbeta

    for layer in reversed(range(num_layers)):
        p = f"layer{layer}."
        ln1_cache, mha_cache, ln2_cache, lin1_cache, gelu_cache, lin2_cache, x_in, x1 = caches[p]

        dx2 = dx
        dffn_out = dx2
        dx1 = dx2.copy()

        dh1_act, dW2, db2 = linear_backward(dffn_out, lin2_cache)
        grads[p + "W2"] += dW2
        grads[p + "b2"] += db2
        dh1 = gelu_backward(dh1_act, gelu_cache)
        dbnorm, dW1, db1 = linear_backward(dh1, lin1_cache)
        grads[p + "W1"] += dW1
        grads[p + "b1"] += db1
        dx1_from_ffn, dln2_gamma, dln2_beta = layernorm_backward(dbnorm, ln2_cache)
        grads[p + "ln2_gamma"] += dln2_gamma
        grads[p + "ln2_beta"] += dln2_beta
        dx1 += dx1_from_ffn

        dattn_out = dx1
        dx_res = dx1.copy()

        da, dWqkv, dWo = mha_backward(dattn_out, mha_cache)
        grads[p + "Wqkv"] += dWqkv
        grads[p + "Wo"] += dWo
        dx_from_attn, dln1_gamma, dln1_beta = layernorm_backward(da, ln1_cache)
        grads[p + "ln1_gamma"] += dln1_gamma
        grads[p + "ln1_beta"] += dln1_beta
        dx_res += dx_from_attn

        dx = dx_res

    token_ids = caches["embed_input"]
    np.add.at(grads["token_emb"], token_ids, dx)
    grads["pos_emb"][: dx.shape[1]] += dx.sum(axis=0)
    return grads


class Adam:
    def __init__(self, params, lr=0.003, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ==========================================================================
# 4. NLP-specific dataset: random crops of the real corpus, and
#    autoregressive text generation with temperature + top-k sampling
# ==========================================================================

def make_batch(encoded_corpus, batch_size, seq_len):
    """Randomly crop (seq_len + 1) windows from the corpus for LM training."""
    max_start = len(encoded_corpus) - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    inputs = np.stack([encoded_corpus[s: s + seq_len] for s in starts])
    targets = np.stack([encoded_corpus[s + 1: s + 1 + seq_len] for s in starts])
    return inputs, targets


def generate_text(params, tokenizer, prompt, max_new_tokens, num_heads, num_layers,
                   max_seq_len, temperature=0.8, top_k=10):
    ids = list(tokenizer.encode(prompt))
    for _ in range(max_new_tokens):
        context = np.array([ids[-max_seq_len:]])  # (1, <=max_seq_len)
        logits, _ = forward(params, context, num_heads, num_layers)
        next_logits = logits[0, -1, :] / temperature

        if top_k is not None:
            top_idx = np.argpartition(next_logits, -top_k)[-top_k:]
            filtered = np.full_like(next_logits, -np.inf)
            filtered[top_idx] = next_logits[top_idx]
            next_logits = filtered

        probs = softmax(next_logits[None, :], axis=-1)[0]
        next_id = rng.choice(len(probs), p=probs)
        ids.append(int(next_id))
    return tokenizer.decode(ids)


# ==========================================================================
# 5. Main: build tokenizer, train the char-level LM, generate text
# ==========================================================================

def main():
    tokenizer = CharTokenizer(CORPUS)
    encoded = tokenizer.encode(CORPUS)
    print(f"Corpus length: {len(CORPUS)} characters")
    print(f"Vocabulary size: {tokenizer.vocab_size} unique characters")
    print(f"Vocabulary: {''.join(tokenizer.chars)!r}\n")

    embed_dim, num_heads, num_layers = 32, 2, 2
    ffn_expansion = 4
    seq_len = 32
    max_seq_len = seq_len
    batch_size = 16
    num_steps = 1500

    params = init_params(tokenizer.vocab_size, embed_dim, num_heads, num_layers, ffn_expansion, max_seq_len)
    optimizer = Adam(params, lr=0.003)

    print("=" * 60)
    print("Training character-level language model on real text")
    print("=" * 60)
    for step in range(1, num_steps + 1):
        inputs, targets = make_batch(encoded, batch_size, seq_len)
        logits, caches = forward(params, inputs, num_heads, num_layers)
        loss, dlogits = cross_entropy_loss(logits, targets)
        grads = backward(params, caches, dlogits, num_heads, num_layers)
        optimizer.step(params, grads)

        if step == 1 or step % 200 == 0:
            print(f"  step {step:5d}  loss={loss:.4f}  perplexity={np.exp(loss):.2f}")

    print("\n" + "=" * 60)
    print("Generated text samples (sampling, not memorized copies)")
    print("=" * 60)
    prompts = ["The old", "She said", "Every ", "The light"]
    for prompt in prompts:
        generated = generate_text(
            params, tokenizer, prompt, max_new_tokens=120,
            num_heads=num_heads, num_layers=num_layers, max_seq_len=max_seq_len,
            temperature=0.7, top_k=8,
        )
        print(f'  prompt: "{prompt}"')
        print(f"  output: {generated!r}\n")


if __name__ == "__main__":
    main()
