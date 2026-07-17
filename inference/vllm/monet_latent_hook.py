"""CapImagine do(Z) latent-intervention hook for the Monet vLLM runner.

Monet feeds the last hidden state of a latent-mode step back as the next input
embedding (the Coconut feedback) in ``monet_gpu_model_runner.py``:

    st["pending"] = last_token_h[i].detach()   # [H]

This module wraps exactly that tensor so we can, at *real generation* time:

  * ``capture``  -- accumulate the empirical mean/std (mu, sigma) of all latent
                    hidden states over a dataset and dump them to disk. This is
                    the "clean pass": the model runs completely unperturbed, so
                    the accuracy measured here is the clean do(Z) baseline.
  * ``corrupt_mean``  -- replace every latent with the global mean mu (collapse;
                    the paper's ``identical`` intervention).
  * ``corrupt_gauss`` -- replace every latent with a fresh draw N(mu, sigma)
                    (destroy; the paper's ``gauss_replace`` intervention).

All corruptions are matched to the empirical latent mu/sigma so there is no
out-of-distribution embedding shock -- an accuracy drop reflects *lost latent
information*, not a foreign vector confusing the model.

Everything is configured through environment variables so it survives the
driver->worker process boundary that vLLM creates:

    MONET_LATENT_MODE   off | capture | corrupt_mean | corrupt_gauss   (default off)
    MONET_LATENT_STATS  path to the mu/sigma stats file (.pt)
    MONET_LATENT_SEED   int seed for the gaussian corruption (default 0)

When the mode is ``off`` (the default, no env set) the hook is a pass-through and
the runner behaves exactly as before.
"""

import os
from typing import Optional

import torch

MODE_OFF = "off"
MODE_CAPTURE = "capture"
MODE_MEAN = "corrupt_mean"
MODE_GAUSS = "corrupt_gauss"
_VALID_MODES = {MODE_OFF, MODE_CAPTURE, MODE_MEAN, MODE_GAUSS}

# Dump the running stats to disk every this many captured latents. The final
# partial window (< this) is dropped, which is negligible for mu/sigma.
_DUMP_EVERY = 500


class MonetLatentHook:
    """Per-worker capture/corruption of the fed-back latent hidden state."""

    def __init__(self,
                 mode: Optional[str] = None,
                 stats_path: Optional[str] = None,
                 seed: Optional[int] = None):
        self.mode = (mode if mode is not None
                     else os.getenv("MONET_LATENT_MODE", MODE_OFF)).strip().lower()
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"MONET_LATENT_MODE={self.mode!r} not in {sorted(_VALID_MODES)}")
        self.stats_path = (stats_path if stats_path is not None
                           else os.getenv("MONET_LATENT_STATS"))
        self.seed = (seed if seed is not None
                     else int(os.getenv("MONET_LATENT_SEED", "0")))

        # capture accumulators (float64 on CPU for numerical stability)
        self._count: int = 0
        self._sum: Optional[torch.Tensor] = None      # [H]
        self._sumsq: Optional[torch.Tensor] = None     # [H]
        self._since_dump: int = 0

        # corruption tensors, lazily moved to the hidden state's device/dtype
        self._mu: Optional[torch.Tensor] = None        # [H]
        self._sigma: Optional[torch.Tensor] = None     # [H]
        self._gen: Optional[torch.Generator] = None

        if self.mode in (MODE_MEAN, MODE_GAUSS):
            self._load_stats()

        print(f"[MonetLatentHook] mode={self.mode} stats={self.stats_path} "
              f"seed={self.seed} active={self.active}")

    @property
    def active(self) -> bool:
        return self.mode != MODE_OFF

    # ------------------------------------------------------------------ stats
    def _load_stats(self):
        if not self.stats_path or not os.path.exists(self.stats_path):
            raise FileNotFoundError(
                f"MONET_LATENT_MODE={self.mode} needs mu/sigma stats, but "
                f"MONET_LATENT_STATS={self.stats_path!r} does not exist. Run the "
                f"clean capture pass first.")
        blob = torch.load(self.stats_path, map_location="cpu")
        self._mu = blob["mu"].to(torch.float32)
        self._sigma = blob["sigma"].to(torch.float32)
        print(f"[MonetLatentHook] loaded stats: mu{tuple(self._mu.shape)} "
              f"sigma{tuple(self._sigma.shape)} from n={blob.get('count')}")

    def _dump_stats(self):
        if not self.stats_path:
            return
        mu = (self._sum / self._count).to(torch.float32)
        var = (self._sumsq / self._count) - mu.double() ** 2
        sigma = var.clamp_min(0).sqrt().to(torch.float32)
        tmp = self.stats_path + ".tmp"
        torch.save({"mu": mu, "sigma": sigma, "count": self._count}, tmp)
        os.replace(tmp, self.stats_path)  # atomic

    def finalize(self):
        """Flush the final capture stats. Safe to call more than once."""
        if self.mode == MODE_CAPTURE and self._count > 0:
            self._dump_stats()
            print(f"[MonetLatentHook] finalized capture: n={self._count} "
                  f"-> {self.stats_path}")

    # ------------------------------------------------------------------- hook
    def process(self, hidden: torch.Tensor) -> torch.Tensor:
        """Wrap the fed-back latent hidden state ``[H]`` for one latent step."""
        if self.mode == MODE_OFF:
            return hidden

        if self.mode == MODE_CAPTURE:
            h = hidden.detach().to(device="cpu", dtype=torch.float64)
            if self._sum is None:
                self._sum = torch.zeros_like(h)
                self._sumsq = torch.zeros_like(h)
            self._sum += h
            self._sumsq += h * h
            self._count += 1
            self._since_dump += 1
            # Write on the very first latent (so a stats file always exists, even
            # on tiny smoke runs) and refresh every _DUMP_EVERY afterwards.
            if self._count == 1 or self._since_dump >= _DUMP_EVERY:
                self._dump_stats()
                self._since_dump = 0
            return hidden  # clean pass: model is unperturbed

        # corruption modes
        mu = self._mu.to(device=hidden.device, dtype=hidden.dtype)
        if self.mode == MODE_MEAN:
            return mu.clone()

        # MODE_GAUSS: N(mu, sigma), fresh draw per latent
        if self._gen is None:
            self._gen = torch.Generator(device="cpu").manual_seed(self.seed)
        sigma = self._sigma.to(dtype=torch.float32)
        noise = torch.randn(sigma.shape, generator=self._gen, dtype=torch.float32)
        corrupted = self._mu.to(torch.float32) + sigma * noise
        return corrupted.to(device=hidden.device, dtype=hidden.dtype)


_HOOK: Optional[MonetLatentHook] = None


def get_hook() -> MonetLatentHook:
    """Process-global hook, built once from the environment."""
    global _HOOK
    if _HOOK is None:
        _HOOK = MonetLatentHook()
    return _HOOK
