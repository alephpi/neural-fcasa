from dataclasses import dataclass
import itertools as it

from einops import repeat

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as fn  # noqa

from einops.layers.torch import Rearrange
from torchaudio.transforms import Spectrogram

from aiaccel.torch.lightning import OptimizerConfig, OptimizerLightningModule


@dataclass
class DumpData:
    logx: torch.Tensor
    lm: torch.Tensor
    z: torch.Tensor
    w: torch.Tensor
    xt: torch.Tensor
    act: torch.Tensor


class AVITask(OptimizerLightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        n_fft: int,
        hop_length: int,
        n_src: int,
        beta: float,
        gamma: float,
        optimizer_config: OptimizerConfig,
        distribution: str = "Gaussian",
        dist_param: dict[str, float] = {},
    ):
        super().__init__(optimizer_config)

        self.encoder = encoder
        self.decoder = decoder

        self.stft = nn.Sequential(
            Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None),
            Rearrange("b m f t -> b f m t"),
        )

        self.hop_length = hop_length
        self.n_src = n_src
        self.beta = beta
        self.gamma = gamma

        perms = torch.tensor(list(it.permutations(range(0, n_src - 1))))
        perms = torch.concat((perms, torch.full((perms.shape[0], 1), n_src - 1, dtype=perms.dtype)), dim=-1)
        self.register_buffer("perms", perms)

        self.distribution = distribution
        self.dist_param = dist_param

    @torch.autocast("cuda", enabled=False)
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx, log_prefix: str = "training"):
        self.dump = None

        wav, act = batch

        # stft
        x = self.stft(wav)[..., : act.shape[-1]]  # [B, F, M, T]
        x /= (xpwr := x.abs().square().clip(1e-6)).mean(dim=(1, 2, 3), keepdims=True).sqrt()
        B, F, M, T = x.shape
        BFT = B * F * T

        # encode
        # qz: latent spectral characteristics
        # qw: speaker activity mask
        # Q: demixing matrix
        # xt: |\tilde{x}_ftm|^2
        # g: diagonal W
        qz, qw, g, Q, xt = self.encoder(x, distribution=True)
        z = qz.rsample()  # [B, D, N, T]
        _, D, *_ = z.shape

        # decode
        # lm: lambda, the PSD of the separated source signals
        lm = self.decoder(z)  # [B, F, N, T]

        # calculate nll
        # add noise channel, set speaker activity mask to 1, to avoid confusing noise with silence.
        act_ = torch.concat([act, torch.ones([B, 1, T], device=act.device)], dim=1)

        act_pit = torch.empty_like(act_)
        pw = qw.probs
        with torch.no_grad():
            # for each sample in a batch, find out the PIT loss
            for b in range(B):
                act_perm_ = act_[b, self.perms]

                yt_ = torch.einsum("pnt,fnt,fmn->pfmt", act_perm_, lm[b], g[b]) + 1e-6
                # normalization trick to make training more stable
                yt_ = yt_ * torch.mean(xt[b].clip(1e-6) / yt_, dim=(1, 2, 3), keepdim=True)
                ratio_xt_yt = xt[b].clip(1e-6) / yt_
                if self.distribution.lower() == "gaussian":
                    nll_x_ = yt_.log().sum(dim=(1, 2, 3)) + ratio_xt_yt.sum(dim=(1, 2, 3))
                elif self.distribution.lower() == "laplace":
                    nll_x_ = yt_.log().sum(dim=(1, 2, 3)) + ratio_xt_yt.sum(dim=2).sqrt().sum(dim=(1, 2))
                elif self.distribution.lower() == "leptokurtic":
                    beta = self.dist_param["beta"]
                    nll_x_ = yt_.log().sum(dim=(1, 2, 3)) + (ratio_xt_yt.sum(dim=2) ** (beta/2)).sum(dim=(1, 2))
                elif self.distribution.lower() == "student-t":
                    nu = self.dist_param["nu"]
                    nll_x_ = yt_.log().sum(dim=(1, 2, 3)) + (nu/2+M) * (torch.log1p((2/nu)*ratio_xt_yt.sum(dim=2)).sum(dim=(1, 2)))
                else:
                    raise ValueError(f"Unsupported distribution: {self.distribution=}, should be one of gaussian, student-t, laplace, leptokurtic.")

                nll_w_ = fn.binary_cross_entropy(
                    repeat(pw[b], "n t -> p n t", p=self.perms.shape[0]),
                    act_perm_,
                    reduction="none",
                ).mean(dim=(1, 2))

                max_indices = (nll_x_ / (F * T) + self.gamma * nll_w_).argmin(dim=0)
                act_pit[b] = act_[b, self.perms[max_indices]]

        del yt_, nll_x_, nll_w_, max_indices

        # signed logdet, add numerical stability
        _, ldQ = torch.linalg.slogdet(Q)  # [B, F]

        # formula under (12), xt is $|\tilde{x}_{ftm}|^2$ in the paper
        yt = torch.einsum("bnt,bfnt,bfmn->bfmt", act_pit, lm, g) + 1e-6
        # normalization trick to make training more stable
        yt = yt * torch.mean(xt.clip(1e-6) / yt, dim=(1, 2, 3), keepdim=True)
        ratio_xt_yt = xt.clip(1e-6) / yt
        if self.distribution.lower() == "gaussian":
            # formula (18)
            nll = yt.log().sum() / BFT + ratio_xt_yt.sum() / BFT - 2 * ldQ.sum() / (B * F)
        elif self.distribution.lower() == "laplace":
            # special case of leptokurtic where beta=1
            nll = yt.log().sum() / BFT + ratio_xt_yt.sum(dim=2).sqrt().sum() / BFT - 2 * ldQ.sum() / (B * F)
        elif self.distribution.lower() == "leptokurtic":
            beta = self.dist_param["beta"]
            nll = yt.log().sum() / BFT + (ratio_xt_yt.sum(dim=2) ** (beta/2)).sum() / BFT - 2 * ldQ.sum() / (B * F)
        elif self.distribution.lower() == "student-t":
            nu = self.dist_param["nu"]
            nll = yt.log().sum() / BFT + (nu/2+M) * (torch.log1p((2/nu)*ratio_xt_yt.sum(dim=2))).sum() / BFT - 2 * ldQ.sum() / (B * F)
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution=}, should be one of gaussian, student-t, laplace, leptokurtic.")

        # calculate kl
        kl = kl_divergence(qz, Normal(0, 1)).sum() / BFT

        nll_w = fn.binary_cross_entropy(qw.probs, act_pit, reduction="mean")

        # calculate loss
        loss = nll + self.beta * kl + self.gamma * nll_w

        # logging
        self.log_dict(
            {
                "step": float(self.trainer.current_epoch),
                f"{log_prefix}/loss": loss,
                f"{log_prefix}/nll": nll,
                f"{log_prefix}/kl": kl,
                f"{log_prefix}/nll_w": nll_w,
            },
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            batch_size=x.shape[0],
            sync_dist=True,
        )

        self.dump = DumpData(
            logx=xpwr[..., 0, :].log().detach(),
            lm=lm.detach(),
            z=qz.mean.detach(),
            w=qw.probs.detach(),
            xt=xt.detach(),
            act=act_pit.detach(),
        )

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx):
        return self.training_step(batch, batch_idx, log_prefix="validation")
