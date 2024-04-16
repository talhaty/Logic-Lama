import torch
from functools import partial
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, prepare_model_for_int8_training 
from utils import TranslationDataPreparer, ContinuousCorrectionDataPreparer, make_parent_dirs
from fol_parser import parse_text_FOL_to_tree
from generate import llama_generate
from torch import cuda, bfloat16
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer
import transformers

# base_model='meta-llama/Llama-2-7b-hf' 
base_model='linhvu/decapoda-research-llama-7b-hf'
prompt_template_path='data/prompt_templates'
load_in_8bit = True
max_output_len = 128
access_token = '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'

#pre-trained
peft_path='yuan-yang/LogicLLaMA-7b-direct-translate-delta-v0'

tokenizer = LlamaTokenizer.from_pretrained(base_model, token=access_token)
tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": '<unk>',
    "pad_token": '<unk>',
})
tokenizer.padding_side = "left"  

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1
)

llama_model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=load_in_8bit,
    torch_dtype=torch.float16,
    device_map='auto',
    token=access_token,
    # quantization_config=bnb_config,
)
llama_model = prepare_model_for_int8_training(llama_model)

model = PeftModel.from_pretrained(
    llama_model,
    peft_path,
    torch_dtype=torch.float16
)
model.to('cuda')


data_preparer = TranslationDataPreparer(
    prompt_template_path,
    tokenizer,
    False,
    256 # just a filler number
)

prepare_input = partial(
    data_preparer.prepare_input,
    **{"nl_key": "NL"},
    add_eos_token=False,
    eval_mode=True,
    return_tensors='pt'
)

simple_generate = partial(
    llama_generate,
    llama_model=model,
    data_preparer=data_preparer,
    max_new_tokens=max_output_len,
    generation_config=generation_config,
    prepare_input=prepare_input,
    return_tensors=False
)

data_point = {'NL': 'The one who created this repo is either a human or an alien'}

full_resp_str, resp_parts = simple_generate(input_str=data_point)

print(full_resp_str)