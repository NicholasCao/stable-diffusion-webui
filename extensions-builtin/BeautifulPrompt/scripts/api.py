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
        model_list = core.model_list()
        return {"model_list": model_list}
    
    @app.post("/beautifulprompt/generate_prompt")
    async def generate_prompt(
        model_name: str = Body(None, title='model_name'),
        raw_prompt: str = Body(None, title='raw_prompt'),
        max_length: int = Body(384, title='max_length'),
        temperature: float = Body(1.0, title='temperature'),
        repetition_penalty: float = Body(1.2, title='repetition_penalty'),
        top_k: int = Body(50, title='top_k'),
        top_p: float = Body(0.95, title='top_p'),
        num_return_sequences: int = Body(5, title='num_return_sequences')
    ):
        try:
            prompts = core.generate_prompts(model_name, raw_prompt, max_length, temperature, repetition_penalty, top_k, top_p, num_return_sequences)
            return {
                'prompts': prompts,
                'success': True
            }
        except Exception as e:
            return {
                'prompts': [],
                'success': False
            }

script_callbacks.on_app_started(beautifulprompt_api)
