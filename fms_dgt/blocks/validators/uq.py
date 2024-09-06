# Standard
from abc import abstractmethod

import torch.nn.functional
from typing import Any
from sentence_transformers.cross_encoder import CrossEncoder

# Local
from fms_dgt.base.block import BaseValidatorBlock
from fms_dgt.base.registry import register_block


def remove_sequence_tokens(toks, logprobs):
    if toks[0] == "<s>":
        toks, logprobs = toks[1:], logprobs[1:]
    if toks[-1] == "</s>":
        toks, logprobs =toks[:-1], logprobs[:-1]
    return toks, logprobs


class UQValidator(BaseValidatorBlock):

    def __init__(self, threshold: float = 0.0,  **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._threshold = threshold

    # prompt is input, rest is output
    # for ease of use toks, logprobs, logits, are the first rows of the topk_ versions
    # topk_ dimensions: num_tokens (in text) x top_k
    @abstractmethod
    def evaluate(self, prompt, text, toks, logprobs, logits, topk_toks, topk_logprobs, topk_logits):
        pass

    def _validate(self, tok_data) -> bool:
        prompt, text, toks, topk_toks, topk_logits = tok_data
        # filter out aux tokens - commented for now since BAM only seems to contain "</s>"
        # and some scorers might want to include that

        # precompute probabilities (approximated based on topk) and extract the ones for all top-1 tokens
        topk_logprobs = torch.nn.functional.log_softmax(topk_logits, dim=-1)

        # NOTE given that we use the tokens we get from BAM atm, toks include special tokens
        uq_score = self.evaluate(prompt=prompt,
                                 text=text,
                                 toks=toks,
                                 logprobs=topk_logprobs[:, 0].squeeze(),
                                 logits=topk_logits[:, 0].squeeze(),
                                 topk_toks=topk_toks,
                                 topk_logprobs=topk_logprobs,
                                 topk_logits=topk_logits)
        # print(self.block_type, uq_score)
        return uq_score > self._threshold


# length-normalized predictive entropy, see eg (8) in
# Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction. ICLR'21
@register_block("ln_pe_scorer")
class LNPE_Validator(UQValidator):
    def evaluate(self, logprobs, **kwargs):
        ln_pe = - 1 / len(logprobs) * sum(logprobs).item()
        return ln_pe


# Jinhao Duan, Hao Cheng, Shiqi Wang, Alex Zavalny, Chenan Wang, Renjing Xu, Bhavya Kailkhura, Kaidi Xu
# Shifting Attention to Relevance:
# Towards the Predictive Uncertainty Quantification of Free-Form Large Language Models, ACL'24
# see also https://github.com/jinhaoduan/SAR, MIT license
@register_block("tok_sar_scorer")
class TokSAR_Validator(UQValidator):

    def __init__(self, measurement_model='cross-encoder/stsb-roberta-large', **kwargs: Any):
        super().__init__(**kwargs)
        assert measurement_model.startswith('cross-encoder')
        self.measure_model = CrossEncoder(model_name=measurement_model, num_labels=1)

    def evaluate(self, prompt, text, toks, logprobs, **kwargs):
        toks, logprobs = remove_sequence_tokens(toks, logprobs)  # wouldn't change anything since don't occur in text
        # measure cosine similarity between output and "removed token in output"
        token_importance = []
        for i, token in enumerate(toks):
            # so they just tokenize to get tokens, but actually they then decode for replacement
            # replace is not ideal since not considering position? (but they use it in paper code as well)
            similarity_to_original = self.measure_model.predict(
                [prompt + text, prompt + text.replace(token, '')])
            token_importance.append(1 - torch.tensor(similarity_to_original))
        # print(token_importance)

        tok_sar = (1 / sum(token_importance) * sum([-p * token_importance[i] for i, p in enumerate(logprobs)])).item()
        return tok_sar
