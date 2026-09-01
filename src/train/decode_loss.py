"""L_dec: auxiliary decoder that reconstructs the observation sentence from the
latents ALONE (writer-side primary loss).

Why CE-decoding fixes collapse where cosine alignment could not: a single mean
vector is "close enough" to every teacher target under cosine, but cannot decode
into thousands of *different* observation sentences under cross-entropy.

Hard constraints (see design doc §4.1, enforced by tests):
  * The decoder's forward takes ONLY (z, target_ids). No image, question, or
    prefix tensors exist in its signature — if it could see the question it would
    write plausible observations from language priors and the gradient into Z
    would vanish.
  * Cross-attention over proj(Z), NOT a rigid slot-i -> token-i map (K=8 latents
    vs 20-30 token sentences; per-position mapping is a documented collapse mode).
  * Small (default 4 layers, d=1024). A large decoder writes fluent observations
    from its own priors while ignoring Z; the loss falls and the latents stay dead.
    The decode-gap metric (shuffled-Z minus real-Z loss) is the tripwire.
  * Discarded at inference. Deployment is unchanged.

Init: token embedding is a seeded Johnson-Lindenstrauss projection of the
backbone's embedding matrix (keeps the language-prior geometry at d_dec); the LM
head is weight-tied to it. Backbone decoder layers are NOT copied — they have no
cross-attention, so surgery would be random-init in disguise; we take honest
random layers plus a prior-preserving embedding instead.
"""
from typing import Optional

import torch
import torch.nn as nn

from .span_nll import find_subsequence  # noqa: F401  (re-export convenience)


class LatentObsDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_latent: int = 3584, d_dec: int = 1024,
                 n_layers: int = 4, n_heads: int = 16, max_len: int = 96,
                 pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.proj = nn.Linear(d_latent, d_dec)
        self.tok = nn.Embedding(vocab_size, d_dec)
        self.pos = nn.Embedding(max_len, d_dec)
        layer = nn.TransformerDecoderLayer(
            d_model=d_dec, nhead=n_heads, dim_feedforward=4 * d_dec,
            batch_first=True, norm_first=True, dropout=0.0)
        self.layers = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_dec)
        # weight-tied head
        self.head = nn.Linear(d_dec, vocab_size, bias=False)
        self.head.weight = self.tok.weight

    @torch.no_grad()
    def init_token_emb_from_backbone(self, backbone_embed_weight: torch.Tensor,
                                     seed: int = 0):
        """JL-project the backbone embedding matrix [V, d_model] down to d_dec."""
        v, d_model = backbone_embed_weight.shape
        g = torch.Generator(device="cpu").manual_seed(seed)
        proj = torch.randn(d_model, self.tok.weight.shape[1], generator=g) / (d_model ** 0.5)
        self.tok.weight.copy_(backbone_embed_weight.float().cpu() @ proj)

    def forward(self, z: torch.Tensor, target_ids: torch.Tensor,
                slot_dropout: int = 0) -> torch.Tensor:
        """CE of reconstructing `target_ids` from latents `z` alone.

        z:          [B, K, d_latent]  (the K generated latents of each sample)
        target_ids: [B, T] observation-span token ids, pad_id-padded.
        slot_dropout: if >0 (training only), randomly hide this many of the K
            slots from cross-attention so no SINGLE slot can be the sole carrier
            (decode-CE forces the block to carry the sentence, but a lone
            informative slot + copies could still decode; this forces the load to
            SPREAD, directly attacking within_block_sim 0.92).
        Returns mean CE over non-pad target tokens.
        """
        B, T = target_ids.shape
        assert T <= self.max_len, f"target len {T} > max_len {self.max_len}"
        K = z.shape[1]
        # Run in the decoder's OWN dtype: the module is cast to the backbone dtype
        # (bf16) at attach time, so a blanket .float() here makes the matmul
        # Float x BFloat16 and the whole loss is silently skipped by the caller's
        # try/except (this shipped: L_dec never ran in jobs 15237561/15427523).
        mem = self.proj(z.to(self.proj.weight.dtype))                 # [B, K, d]
        mem_key_padding = None
        if self.training and slot_dropout > 0 and slot_dropout < K:
            # hide `slot_dropout` random slots per sample (True = ignore)
            mask = torch.zeros(B, K, dtype=torch.bool, device=mem.device)
            for b in range(B):
                drop = torch.randperm(K, device=mem.device)[:slot_dropout]
                mask[b, drop] = True
            mem_key_padding = mask
        # teacher forcing: input is <shifted> targets; BOS is a zeros row.
        tok = self.tok(target_ids.clamp_min(0))
        tok = torch.cat([torch.zeros_like(tok[:, :1]), tok[:, :-1]], dim=1)
        x = tok + self.pos(torch.arange(T, device=tok.device))[None]
        # dtype= matters: the mask defaults to fp32 and is ADDED to bf16 attention
        # scores (the next dtype error after the proj one).
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=x.device, dtype=x.dtype)
        h = self.layers(x, mem, tgt_mask=causal, memory_key_padding_mask=mem_key_padding)
        logits = self.head(self.norm(h))                              # [B, T, V]
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            target_ids.reshape(-1),
            ignore_index=self.pad_id, reduction="mean")
        return loss

    @torch.no_grad()
    def decode_gap(self, z: torch.Tensor, target_ids: torch.Tensor,
                   donor: Optional[torch.Tensor] = None) -> float:
        """L_dec(other Z) − L_dec(real Z). Near zero ⇒ the decoder is writing from
        language priors and the writer objective is a no-op.

        The comparison Z is, in order of preference: an explicit `donor` (a real
        other-sample latent block — the interchange-style control), a batch roll
        (B>1), or matched-moment noise.

        B=1 CAVEAT (this shipped broken in jobs 15237561/15427523/15439209):
        Stage 3 trains at bsz=1, where `torch.roll(z, 1, dims=0)` on a size-1 axis
        is a NO-OP, so the gap was identically 0.0 and the tripwire read nothing.
        Shuffling slots instead is also a no-op here: cross-attention over the
        memory carries no positional encoding, so the decoder is permutation-
        invariant across the K slots.
        """
        real = self.forward(z, target_ids).item()
        alt = None
        if donor is not None:
            alt = donor.to(z.device, z.dtype)
            if alt.shape != z.shape:
                alt = None
        if alt is None:
            alt = (torch.roll(z, shifts=1, dims=0) if z.shape[0] > 1
                   else torch.randn_like(z) * z.std() + z.mean())
        return self.forward(alt, target_ids).item() - real
