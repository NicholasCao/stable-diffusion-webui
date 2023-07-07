import html
import os
import time

import requests
import torch
import transformers

from modules import shared, generation_parameters_copypaste

from modules import scripts, script_callbacks, devices, ui, paths
import gradio as gr

from modules.ui_components import FormRow


NEGATIVE_PROMPT = \
'(NSFW), (((ugly))), ((duplicate)), (morbid), (mutilated), [out of frame], (extra fingers), mutated hands, \
(poorly drawn hands), (mutation), (deformed), blurry, (bad anatomy), \
(bad proportions), (extra limbs), (disfigured), gross proportions, (malformed limbs), \
(missing arms), (missing legs), (extra arms), (extra legs), (fused fingers), (too many fingers), (long neck)'

class Model:
    name = None
    model = None
    tokenizer = None


available_models = []
current = Model()

if shared.cmd_opts.beautifulprompt_dir is not None:
    models_dir = shared.cmd_opts.beautifulprompt_dir
elif hasattr(shared.cmd_opts, 'public_cache') and shared.cmd_opts.public_cache:
    models_dir = '/stable-diffusion-cache/models/BeautifulPrompt'
else:
    models_dir = os.path.join(paths.models_path, "BeautifulPrompt")

def device():
    return devices.cpu if shared.opts.beautifulprompt_device == 'cpu' else devices.device


def model_list():
    available_models = []

    os.makedirs(models_dir, exist_ok=True)
    
    for dirname in os.listdir(models_dir):
        if os.path.isdir(os.path.join(models_dir, dirname)):
            available_models.append(dirname)
    
    return available_models


def get_model_path(name):
    dirname = os.path.join(models_dir, name)
    if not os.path.isdir(dirname):
        return name

    return dirname


def generate_batch(input_ids, max_length, temperature, repetition_penalty, top_k, top_p, num_return_sequences):
    top_p = float(top_p)
    top_k = int(top_k)

    outputs = current.model.generate(
        input_ids,
        do_sample=True,
        temperature=max(float(temperature), 1e-6),
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=current.tokenizer.pad_token_id or current.tokenizer.eos_token_id
    )

    texts = current.tokenizer.batch_decode(outputs[:, input_ids.size(1):], skip_special_tokens=True)
    texts = [text.strip() for text in texts]
    return texts

def generate_prompts(
        model_name: str,
        raw_prompt: str,
        max_length: int,
        temperature: float,
        repetition_penalty: float,
        top_k: int,
        top_p: float,
        num_return_sequences: int
    ):

    if current.name != model_name:
        shared.state.textinfo = "Loading model..."
        current.tokenizer = None
        current.model = None
        current.name = None

        if model_name != 'None':
            path = get_model_path(model_name)
            current.tokenizer = transformers.AutoTokenizer.from_pretrained(path)
            current.model = transformers.AutoModelForCausalLM.from_pretrained(path)
            current.name = model_name

            # is full precision
            if all(param.dtype == torch.float32 for param in current.model.parameters()):
                current.model.half()

    assert current.model, 'No model available'
    assert current.tokenizer, 'No tokenizer available'

    shared.state.textinfo = "Generating prompts..."

    current.model.to(device())
    
    text = f'Instruction: Give a simple description of the image to generate a drawing prompt.\nInput: {raw_prompt}\nOutput:'

    input_ids = current.tokenizer(text, return_tensors="pt").input_ids
    if input_ids.shape[1] == 0:
        input_ids = torch.asarray([[current.tokenizer.bos_token_id]], dtype=torch.long)
    input_ids = input_ids.to(device())

    prompts = generate_batch(input_ids, max_length, temperature, repetition_penalty, top_k, top_p, num_return_sequences)
    
    return prompts


def model_selection_changed(model_name):
    if model_name in ["API", "None"]:
        current.tokenizer = None
        current.model = None
        current.name = None

        devices.torch_gc()

def request_api(api_url, api_token, raw_prompt, max_length, temperature, repetition_penalty, top_k, top_p, num_return_sequences):
    data = {
        'raw_prompt': raw_prompt,
        'max_length': max_length,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,
        'top_k': top_k,
        'top_p': top_p,
        'num_prompts': num_return_sequences
    }
    if api_token is not None:
        headers = {'Authorization': api_token}

    response = requests.post(api_url, json=data, headers=headers)
    response = response.json()
    
    return response['prompts']
