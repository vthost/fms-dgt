# Standard
import numpy as np
from typing import Any, List
import copy
import os
import torch

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.instance import Instance
from fms_dgt.base.registry import get_resource, register_block
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.resources.genai import GenAIKeyResource
import fms_dgt.blocks.generators.utils as generator_utils

try:
    # Third Party
    from dotenv import load_dotenv
    from genai import Client, Credentials
    from genai.schema import (
        TextGenerationParameters,
        TextGenerationReturnOptions,
        TextTokenizationParameters,
        TextTokenizationReturnOptions,
    )
except ModuleNotFoundError:
    pass


@register_block("genai")
class GenAIGenerator(LMGenerator):
    """GenAI Generator"""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        try:
            # Third Party
            import genai  # noqa: E401
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'genai' LM type, but package `genai` not installed. ",
                "please install these via `pip install -r fms_dgt[genai]`",
            )

        self._genai_resource: GenAIKeyResource = get_resource("genai", "GENAI_KEY")

        load_dotenv()
        credentials = Credentials(
            self._genai_resource.key, api_endpoint=os.getenv("GENAI_API", None)
        )
        self.client = Client(credentials=credentials)

    def generate_batch(
        self, requests: List[Instance], disable_tqdm: bool = False, return_tokens: bool = False
    ) -> None:
        # group requests by kwargs
        grouper = generator_utils.Grouper(requests, lambda x: str(x.kwargs))
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_batch requests",
        )

        for key, reqs in grouper.get_grouped().items():
            chunks: List[List[Instance]] = generator_utils.chunks(
                reqs, n=self._genai_resource.max_calls
            )

            for chunk in chunks:
                inputs = [instance.args[0] for instance in chunk]
                # all kwargs are identical within a chunk
                gen_kwargs = next(iter(chunk)).kwargs

                if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                    # start with default params then overwrite with kwargs
                    kwargs = {**self._base_kwargs, **kwargs}
                else:
                    raise ValueError(
                        f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                    )
                model_id = kwargs.pop("model_id_or_path", self.model_id_or_path)
                until = kwargs.get("stop_sequences", None)

                top_k = 5  # max allowed in BAM, are needed to calculate/approximate probabilities wrt vocabulary
                parameters = TextGenerationParameters(
                    return_options=TextGenerationReturnOptions(
                        input_text=True,
                        generated_tokens=True,
                        top_n_tokens=top_k,
                        token_logprobs=True,
                    ),
                    **kwargs,
                )

                responses = list(
                    self.client.text.generation.create(
                        model_id=model_id,
                        inputs=inputs,
                        parameters=parameters,
                    )
                )

                for instance in chunk:
                    result = next(
                        resp.results[0]
                        for resp in responses
                        if instance.args[0] == resp.results[0].input_text
                    )

                    s = result.generated_text
                    if return_tokens:
                        # provide both just for simplicity
                        tokens = [t.text for t in result.generated_tokens]
                        topk_tokens = np.ndarray((len(result.generated_tokens), top_k), dtype=object)
                        topk_logits = torch.zeros(len(result.generated_tokens), top_k)
                        for i, t in enumerate(result.generated_tokens):
                            for j in range(top_k):
                                if t.top_tokens[j].logprob is not None:
                                    topk_tokens[i, j] = t.top_tokens[j].text
                                    topk_logits[i, j] = t.top_tokens[j].logprob
                        s = (s, tokens, topk_tokens, topk_logits)

                    self.update_instance_with_result(
                        "generate_batch",
                        s,
                        instance,
                        until,
                    )
                    pbar.update(1)

        pbar.close()

    def loglikelihood_batch(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> None:
        # group requests by kwargs
        grouper = generator_utils.Grouper(requests, lambda x: str(x.kwargs))
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood_batch requests",
        )

        for key, reqs in grouper.get_grouped().items():
            chunks: List[List[Instance]] = generator_utils.chunks(
                reqs, n=self._genai_resource.max_calls
            )

            for chunk in chunks:
                to_score = ["".join(instance.args) for instance in chunk]
                to_tokenize = [instance.args[-1] for instance in chunk]
                # all kwargs are identical within a chunk
                gen_kwargs = next(iter(chunk)).kwargs

                if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                    # start with default params in self.config then overwrite with kwargs
                    kwargs = {**self._base_kwargs, **kwargs}
                else:
                    raise ValueError(
                        f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                    )
                model_id = kwargs.pop("model_id_or_path", self.model_id_or_path)

                score_params = TextGenerationParameters(
                    temperature=1.0,
                    decoding_method="greedy",
                    max_new_tokens=1,
                    min_new_tokens=0,
                    return_options=TextGenerationReturnOptions(
                        generated_tokens=True,
                        token_logprobs=True,
                        input_text=True,
                        input_tokens=True,
                    ),
                )

                score_responses = list(
                    self.client.text.generation.create(
                        model_id=model_id,
                        inputs=to_score,
                        parameters=score_params,
                    )
                )

                tok_responses = next(
                    self.client.text.tokenization.create(
                        model_id=model_id,
                        input=to_tokenize,
                        parameters=TextTokenizationParameters(
                            return_options=TextTokenizationReturnOptions(
                                tokens=True,
                                input_text=True,
                            ),
                        ),
                    )
                ).results

                for instance in chunk:
                    score_result = next(
                        resp.results[0]
                        for resp in score_responses
                        if "".join(instance.args) == resp.results[0].input_text
                    )
                    tok_count = next(
                        resp.token_count
                        for resp in tok_responses
                        if instance.args[-1] == resp.input_text
                    )

                    s = score_result.input_tokens
                    # tok_ct - 1 since first token in encoding is bos
                    s_toks = s[-(tok_count - 1) :]

                    answer = sum(
                        [tok.logprob for tok in s_toks if tok.logprob is not None]
                    )

                    self.update_instance_with_result(
                        "loglikelihood_batch", answer, instance
                    )
                    pbar.update(1)

        pbar.close()
