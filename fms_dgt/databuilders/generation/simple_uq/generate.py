# Standard
from typing import Any, Dict, List, Tuple
import copy
import random
import time

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.base.task import SdgTask, group_data_by_task
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.blocks.validators.rouge import RougeDedupValidator
from fms_dgt.blocks.validators.uq import UQValidator
from fms_dgt.databuilders.generation.simple.task import (
    InstructLabSdgData,
    InstructLabSdgTask,
)
from fms_dgt.utils import sdg_logger
import fms_dgt.databuilders.generation.simple_uq.utils as utils


@register_data_builder("simple_uq")
class SimpleInstructDataBuilder(DataBuilder):
    """Class for InstructLab"""

    TASK_TYPE: SdgTask = InstructLabSdgTask

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator

    # val1 is the validator which checks rouge score
    val1: RougeDedupValidator

    # val1 is the validator which checks given uncertainty score
    val2: UQValidator

    def __init__(
        self,
        *args: Any,
        num_prompt_instructions: int = 2,
        prompt_file_path: str = "prompt.txt",
        request_batch_size: int = 5,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self._prompt_template = utils.check_prompt_file(
            prompt_file_path, self.llm1.model_id_or_path
        )
        self._num_prompt_instructions = num_prompt_instructions
        self._request_batch_size = request_batch_size

    def _encode_prompt(self, prompt_instructions, num_tasks):
        # defining this as its own separate method allows us to overwrite it for subclasses
        prompt = utils.encode_prompt(prompt_instructions, self._prompt_template, num_tasks)
        return prompt

    def __call__(
        self,
        request_idx: int,
        instruction_data: List[InstructLabSdgData],
    ) -> List[InstructLabSdgData]:

        sdg_logger.info(f"SIMPLE GENERATE request {request_idx}")
        inputs: List[Dict] = []
        # the seed examples
        instruction_data = instruction_data + []
        random.shuffle(instruction_data)
        # create prompts with self._num_prompt_instructions examples until latter are used up
        for grouped_data in group_data_by_task(instruction_data):
            for i in range(0, len(grouped_data), self._num_prompt_instructions):
                prompt_instructions = grouped_data[
                    i : i + self._num_prompt_instructions
                ]

                prompt = self._encode_prompt(prompt_instructions, len(prompt_instructions) + 1)
                inp = {
                    "prompt": prompt,
                    "stop_sequences": [f"* Task {len(prompt_instructions)+2}"],
                    "data": prompt_instructions,
                }
                inputs.append(inp)

        request_start = time.time()

        llm_outputs = self.llm1.generate(inputs, return_tokens=True)
        request_duration = time.time() - request_start

        post_process_start = time.time()
        llm_data: List[InstructLabSdgData] = []
        llm_data_tok: List[Tuple] = []
        for gen_inp in llm_outputs:
            prompt_instructions: List[InstructLabSdgData] = gen_inp["data"]  # list each containing a seed example
            # splits the generated text into individual instructions
            # VT we would need to split logprobs/tokens and therefore restrict generations to ONE task
            #  (see the adapted prompt template)
            #  discarding all but first doesn't work since we don't know where to start discarding logprobs/tokens
            new_instruction_dicts, discarded = utils.post_process_gpt3_response(
                len(prompt_instructions),
                gen_inp["output"][0],
            )

            # make sure the generated instruction carried over extra fields
            for new_ins_dict, orig_ins in zip(
                new_instruction_dicts, prompt_instructions
            ):
                new_ins = copy.copy(orig_ins)
                new_ins.instruction = new_ins_dict.get("instruction")
                new_ins.input = new_ins_dict.get("input")
                new_ins.output = new_ins_dict.get("output")
                llm_data.append(new_ins)

            # input for uq validator
            assert len(new_instruction_dicts) < 2
            if new_instruction_dicts:
                # create one tuple with all inputs for validator
                validator_input = (gen_inp["prompt"],)
                validator_input += gen_inp["output"]
                llm_data_tok.append(validator_input)

        post_process_duration = time.time() - post_process_start
        sdg_logger.info(
            "Request %s took %.2fs, post-processing took %.2fs",
            request_idx,
            request_duration,
            post_process_duration,
        )

        assess_start = time.time()

        # VT cannot run rouge validator atm: we need indices from what got discarded by rouge validator
        #  so need either auxiliary attributes in InstructLabSdgData to carry around the token data
        #   (ie the input for uq validator)
        #  or (rouge) validator to return which indices it kept/discarded
        #  or a composite validator which manages chaining validators
        # # now we assess and filter with rouge
        # all_instructions = [instr.instruction for instr in instruction_data]
        #
        # val_inputs: List[InstructLabSdgData] = []
        # for instruction_data_entry in llm_data:
        #     # computing similarity with the pre-tokenized instructions
        #     inp = {
        #         "to_check": instruction_data_entry.instruction,
        #         "data": instruction_data_entry,
        #     }
        #     val_inputs.append(inp)
        #
        # # filter rouge data
        # outputs = [
        #     output["data"]
        #     for output in self.val1.generate(
        #         val_inputs,
        #         context=all_instructions,
        #         arg_fields=["to_check"],
        #         result_field="output",
        #     )
        # ]

        val_inputs = [{
                "instruction_data": entry,  # just to carry it with us and be able to use it as output later
                "tok_data": llm_data_tok[ie],
        } for ie, entry in enumerate(llm_data)]

        outputs = self.val2.generate(
            val_inputs,               # for base validator
            arg_fields=["tok_data"],  # this will be given to uq validator
            result_field="output",
        )

        outputs = [o["instruction_data"] for o in outputs]

        discarded += len(llm_data) - len(outputs)

        assess_duration = time.time() - assess_start
        sdg_logger.info(
            "Assessing generated samples took %.2fs, discarded %s instances",
            assess_duration,
            discarded,
        )

        return outputs
