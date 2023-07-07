import html
import os
import time

import requests
import torch
import transformers

from fastapi import FastAPI, Body
from modules import shared, generation_parameters_copypaste

from modules import scripts, script_callbacks, devices, ui, paths
import gradio as gr

from modules.ui_components import FormRow

from scripts import core

def beautifulprompt_api(_: gr.Blocks, app: FastAPI):

    @app.get("/beautifulprompt/model_list")
    async def model_list():
        return {"model_list": core.model_list}
    
    @app.post("/beautifulprompt/generate_prompt")
    async def generate_prompt(
        model_name: str,
        raw_prompt: str,
        max_length: int,
        temperature: float,
        repetition_penalty: float,
        top_k: int,
        top_p: float,
        num_return_sequences: int
    ):
        try:
            return {
                'prompts': core.generate_prompts(model_name, raw_prompt, max_length, temperature, repetition_penalty, top_k, top_p, num_return_sequences),
                'success': True
            }
        except:
            return {
                'prompts': [],
                'success': False
            }

try:
    script_callbacks.on_app_started(beautifulprompt_api)
except:
    pass
