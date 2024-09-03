# Standard
import math

import torch.nn.functional
from typing import Any

# Local
from fms_dgt.base.block import BaseValidatorBlock
from fms_dgt.base.registry import register_block


@register_block("uq_scorer")
class UQValidator(BaseValidatorBlock):

    def __init__(self, threshold: float = 1.1, metric: str = "ln_pe", **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if threshold is None:
            # if threshold is set to None, we'll put it as an unreachable (it's used as < threshold) high value
            threshold = 1.1

        self._threshold = threshold
        self._metric = metric  # later: adapt computation to given metric
        assert self._metric == "ln_pe"

    def _validate(self, tok_data) -> bool:
        # text, toks, logits are LLM outputs
        # toks: num_tokens (in text) x top_k
        # logits: same dimensions
        prompt, text, topk_toks, topk_logits = tok_data

        if topk_toks[0] == "<s>":
            topk_toks, topk_logits = topk_toks[1:], topk_logits[1:]
        if topk_toks[-1] == "</s>":
            topk_toks, topk_logits = topk_toks[:-1], topk_logits[:-1]

        # compute probabilities (approximated based on topk) and extract the ones for all top-1 tokens
        logprobs = torch.nn.functional.log_softmax(topk_logits, dim=-1)[:, 0].squeeze()

        # length-normalized predictive entropy
        # see Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction.
        # NOTE given that we use BAM atm, this includes special tokens
        ln_pe = - 1/len(logprobs) * sum(logprobs).item()
        # print("LN-PE", ln_pe)

        return ln_pe < self._threshold
