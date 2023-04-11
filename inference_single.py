from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
from transformers import GenerationConfig

from train import ModelArguments, smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, \
  DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, PROMPT_DICT


# SYSTEM_PROMPT = """You are a succinct and helpful assistant.
#
# Your job is to answer an action that is morally wrong given image caption, and give an explanation about the judgment. Then, you have to answer another image caption to make the action morally plausible, and give an explanation about the judgment again.
#
# For example,
# If I give you a caption as "Caption: a statue on a river next to buildings and a water fountain", your answer should be:
# Action: Throwing trash
# Judgment: It's wrong
# Explanation: The statue is a cultural heritage, and it is not okay to throw trash near it.
#
# Caption: a disposal can
# Action: Throwing trash
# Judgment: It's okay
# Explanation: The disposal can is for throwing trash, and it is okay to throw trash near it.
# ---
# I gave the caption "a statue on a river next to buildings and a water fountain". Then, you are supposed to generate an action morally wrong, "Throwing trash".
# Next, You have to generate a caption to make the action you generated action "Throwing trash" morally plausible.
#
# Requirements:
# - You should use the exactly same action to answer.
# - Do not change even one character of the action first you answered.
# - Your answer should follow the given form of the example. Given the caption, you should generate action and judgment saying that is wrong, and caption, action, and judgment saying that is okay."""\
#
# USER_PROMPT = """Think carefully about the requirements.
# Caption: {situation}
# Action:
# Judgment:
# Explanation:"""

SYSTEM_PROMPT = """Your job is to answer an action that is morally wrong given image caption, and give an explanation about the judgment. Then, you have to answer another image caption to make the action morally plausible, and give an explanation about the judgment again."""

USER_PROMPT = """Think carefully about the requirements.
Caption: {situation}"""

@dataclass
class InferenceArguments:
  model_max_length: int = field(
    default=512,
    metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
  )
  load_in_8bit: bool = field(
    default=False,
    metadata={"help": "Load the model in 8-bit mode."},
  )
  inference_dtype: torch.dtype = field(
    default=torch.float32,
    metadata={"help": "The dtype to use for inference."},
  )


def generate_prompt(instruction, input=None):
  if input:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def inference():
  parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
  model_args, inference_args = parser.parse_args_into_dataclasses()

  model = transformers.AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    load_in_8bit=inference_args.load_in_8bit,
    torch_dtype=inference_args.inference_dtype,
    device_map="auto",
  )
  model.cuda()
  model.eval()

  generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
    max_length=inference_args.model_max_length,
    min_new_tokens=16,
  )

  tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    use_fast=False,
    model_max_length=inference_args.model_max_length,
  )

  if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
      special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
      tokenizer=tokenizer,
      model=model,
    )
  tokenizer.add_special_tokens(
    {
      "eos_token": DEFAULT_EOS_TOKEN,
      "bos_token": DEFAULT_BOS_TOKEN,
      "unk_token": DEFAULT_UNK_TOKEN,
    }
  )

  ctx = ""
  instruction = SYSTEM_PROMPT
  for input in [
    "a collage of people posing with a cake",
    "mosaic of the crucifixion of jesus",
    "a building with a clock on top of it",
  ]:
    input = USER_PROMPT.format(situation=input)
    inputs = tokenizer(generate_prompt(instruction, input), return_tensors="pt")
    print(f"[INPUT]\n{input}")
    outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
                             generation_config=generation_config,
                             max_new_tokens=inference_args.model_max_length,
                             return_dict_in_generate=True,
                             output_scores=True)
    # transition_scores = model.compute_transition_scores(
    #   outputs.sequences, outputs.scores, normalize_logits=True
    # )
    # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
    # encoder-decoder models, like BART or T5.
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]

    # for tok, score in zip(generated_tokens[0], transition_scores[0]):
    #   # | token | token string | logits | probability
    #   print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy()):.2%}")
    ctx += f"Instruction: {instruction}\n" + f"Response: {generated_tokens[0]}\n"
    print("[Response]\n", tokenizer.decode(generated_tokens[0]))
    print()


if __name__ == "__main__":
  inference()
