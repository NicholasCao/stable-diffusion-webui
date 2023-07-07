import html
import os
import time
import json

import requests
import torch
import transformers
import requests

from modules import shared, generation_parameters_copypaste

from modules import scripts, script_callbacks, devices, ui, paths
import gradio as gr

from modules.ui_components import FormRow

from scripts import core

base_dir = scripts.basedir()

available_models = []

def list_available_models():
    global available_models
    available_models.clear()

    if hasattr(shared.cmd_opts, 'just_ui') and shared.cmd_opts.just_ui:
        req_url = '/'.join([shared.cmd_opts.server_path, 'beautifulprompt/model_list'])
        try:
            data = requests.get(req_url)
            if data.status_code == 200:
                available_models = json.loads(data.text)['model_list']
            else:
                print(data.status_code, data.text)
        except Exception as e:
            print(e)
        
    else:
        available_models = core.model_list()
        for name in [x.strip() for x in shared.opts.beautifulprompt_names.split(",")]:
            if not name:
                continue

            available_models.append(name)
    available_models.append('API')
    print(available_models)


def generate(id_task, model_name, api_url, api_token, raw_prompt, max_length, temperature, repetition_penalty, top_k, top_p, num_return_sequences):
    shared.state.job_count = 1

    if model_name == "API":
        # use custom api
        shared.state.textinfo = "Generating prompts via api..."
        prompts = core.request_api(api_url, api_token, raw_prompt, max_length, temperature, repetition_penalty, top_k, top_p, num_return_sequences)
    else:
        if hasattr(shared.cmd_opts, 'just_ui') and shared.cmd_opts.just_ui:
            shared.state.textinfo = "Generating prompts via api..."
            req_url = '/'.join([shared.cmd_opts.server_path, 'beautifulprompt/generate_prompt'])
            prompts = core.request_api(req_url, None, raw_prompt, max_length, temperature, repetition_penalty, top_k, top_p, num_return_sequences)
        
        else:
            prompts = core.generate_prompts(model_name, raw_prompt, max_length, temperature, repetition_penalty, top_k, top_p, num_return_sequences)
        

    markup = '<table><tbody>'

    index = 0
    shared.state.nextjob()
    for generated_prompt in prompts:
        index += 1
        markup += f"""
<tr>
<td class="prompt-td">
<div class="prompt gr-box gr-text-input">
    <p id='beautifulprompt_res_{index}'>{html.escape(generated_prompt)}</p>
</div>
</td>
<td class="sendto">
    <button class='gr-button gr-button-lg gr-button-secondary' onclick="send_to_txt2img(gradioApp().getElementById('beautifulprompt_res_{index}').textContent)">to txt2img</a>
    <button class='gr-button gr-button-lg gr-button-secondary' onclick="send_to_img2img(gradioApp().getElementById('beautifulprompt_res_{index}').textContent)">to img2img</a>
</td>
</tr>
"""

    markup += '</tbody></table>'

    return markup, ''


def find_prompts(fields):
    field_prompt = [x for x in fields if x[1] == "Prompt"][0]
    field_negative_prompt = [x for x in fields if x[1] == "Negative prompt"][0]
    return [field_prompt[0], field_negative_prompt[0]]


def send_prompts(text, also_generate_negative_prompt):
    params = generation_parameters_copypaste.parse_generation_parameters(text)
    negative_prompt = core.NEGATIVE_PROMPT if also_generate_negative_prompt else ''
    return params.get("Prompt", ""), negative_prompt or gr.update()


def add_tab():
    list_available_models()

    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row():
            with gr.Column(scale=80):
                prompt = gr.Textbox(label="Prompt", elem_id="beautifulprompt_prompt", show_label=False, lines=2, placeholder="Beginning of the prompt (press Ctrl+Enter or Alt+Enter to generate)").style(container=False)
            with gr.Column(scale=10):
                submit = gr.Button('Generate', elem_id="beautifulprompt_generate", variant='primary')

        with gr.Row(elem_id="beautifulprompt_main"):
            with gr.Column(variant="compact"):
                selected_text = gr.TextArea(elem_id='beautifulprompt_selected_text', visible=False)
                send_to_txt2img = gr.Button(elem_id='beautifulprompt_send_to_txt2img', visible=False)
                send_to_img2img = gr.Button(elem_id='beautifulprompt_send_to_img2img', visible=False)

                with FormRow():
                    model_selection = gr.Dropdown(label="Model", elem_id="beautifulprompt_model", value=available_models[0], choices=available_models + ["None"])
                
                with FormRow(elem_id='beautifulprompt_api', visible=False):
                    api_url = gr.Textbox(elem_id='beautifulprompt_api_url', label="API URL")
                    api_token = gr.Textbox(elem_id='beautifulprompt_api_token', label="API Token")

                with FormRow():
                    # sampling_mode = gr.Radio(label="Sampling mode", elem_id="beautifulprompt_sampling_mode", value="Top K", choices=["Top K", "Top P"])
                    top_k = gr.Slider(label="Top K", elem_id="beautifulprompt_top_k", value=50, minimum=1, maximum=300, step=1)
                    top_p = gr.Slider(label="Top P", elem_id="beautifulprompt_top_p", value=0.95, minimum=0, maximum=1, step=0.01)

                with gr.Row():
                    temperature = gr.Slider(label="Temperature", elem_id="beautifulprompt_temperature", value=1.0, minimum=0.05, maximum=3, step=0.05)
                    repetition_penalty = gr.Slider(label="Repetition penalty", elem_id="beautifulprompt_repetition_penalty", value=1.1, minimum=1, maximum=2, step=0.05)

                with FormRow():
                    max_length = gr.Slider(label="Max length", elem_id="beautifulprompt_max_length", value=384, minimum=1, maximum=400, step=1)
                    num_return_sequences = gr.Slider(label="Num prompts", elem_id="beautifulprompt_num_return_sequences", value=5, minimum=1, maximum=50, step=1)
                with FormRow(variant="compact"):
                    also_generate_negative_prompt = gr.Checkbox(label='Negative prompt', value=True, elem_id="beautifulprompt_also_generate_negative_prompt")

                with open(os.path.join(base_dir, "explanation.html"), encoding="utf8") as file:
                    footer = file.read()
                    gr.HTML(footer)

            with gr.Column():
                with gr.Group(elem_id="beautifulprompt_results_column"):
                    res = gr.HTML()
                    res_info = gr.HTML()

        submit.click(
            fn=ui.wrap_gradio_gpu_call(generate, extra_outputs=['']),
            _js="submit_prompt",
            inputs=[model_selection, model_selection, api_url, api_token, prompt, max_length, temperature, repetition_penalty, top_k, top_p, num_return_sequences],
            outputs=[res, res_info]
        )

        model_selection.change(
            fn=core.model_selection_changed,
            _js="show_or_hide",
            inputs=[model_selection],
            outputs=[],
        )

        send_to_txt2img.click(
            fn=send_prompts,
            inputs=[selected_text, also_generate_negative_prompt],
            outputs=find_prompts(ui.txt2img_paste_fields)
        )

        send_to_img2img.click(
            fn=send_prompts,
            inputs=[selected_text, also_generate_negative_prompt],
            outputs=find_prompts(ui.img2img_paste_fields)
        )

    return [(tab, "BeautifulPrompt", "beautifulprompt")]


def on_ui_settings():
    section = ("beautifulprompt", "BeautifulPrompt")

    shared.opts.add_option("beautifulprompt_names", shared.OptionInfo("alibaba-pai/pai-bloom-1b1-text2prompt-sd", "Hugginface model names for Beautiful Prompt, separated by comma", section=section))
    shared.opts.add_option("beautifulprompt_device", shared.OptionInfo("gpu", "Device to use for text generation", gr.Radio, {"choices": ["gpu", "cpu"]}, section=section))


def on_unload():
    core.current.model = None
    core.current.tokenizer = None


script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_script_unloaded(on_unload)
