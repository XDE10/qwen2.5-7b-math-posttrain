"""Versioned TRL alignment: v1 formulas plus empty truncated rows and KL."""
from src.losses.reference import group_advantages, policy_loss


def trl_policy_loss(logp, old_logp, advantages, mask, mode, constant, beta=0., ref_logp=None, accumulation=1, bias_correction_kl=True):
    # TRL leaves fully truncated rows in the batch denominator with zero loss.
    active = mask.sum(-1) > 0
    if not active.any():
        return logp.sum()*0
    value = policy_loss(logp[active], old_logp[active], advantages[active], mask[active], mode=mode, constant=constant)
    value = value * active.float().mean()
    if beta:
        kl = (ref_logp-logp).exp() - (ref_logp-logp) - 1
        if bias_correction_kl:
            kl = kl * (logp-old_logp.detach()).exp()
        total = (kl*mask).sum(-1)
        term = (total/mask.sum(-1).clamp(min=1)).mean() if mode=='grpo' else (total/constant).mean()
        value = value + beta*term
    return value/accumulation
