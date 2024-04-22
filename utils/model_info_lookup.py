import gradio as gr
import json
from transformers import AutoTokenizer
from vocab_utils import get_vocab_family, get_tokenizer_sha

model_history = {}

try:
    with open("model_history.json", "r") as file:
        model_history = json.load(file)
except FileNotFoundError:
    pass

def save_history():
    with open("model_history.json", "w") as file:
        json.dump(model_history, file, indent=4)

def update_history_display():
    entries = []
    for key, value in model_history.items():
        if "error" in value:
            entries.append(f"{key}: {value['error']}")
            continue

        vocab_family = value.get("vocab_family", "Unknown")
        tokenizer_sha = value.get("tokenizer_sha", None)

        if vocab_family == "Unknown":
            entries.append(f"{key}: {vocab_family}, Tokenizer SHA: {tokenizer_sha}, Likely vocab family: {value['tokenizer_class']}")
        else:
            entries.append(f"{key}: {vocab_family}")

    return "\n".join(entries)

def get_vocab_info(request, branch="main"):
    if not request.strip():
        return None, None, None

    if request in model_history:
        vocab_info = model_history[request]
        model_name = request.split(' ')[0]
        branch = request.split(' ')[1][1:-1]
        return vocab_info, model_name, branch
    
    parts = request.split('/')
    model_name = '/'.join(parts[-2:])  # model_creator/model_name

    if 'tree' in parts:
        branch = parts[parts.index('tree') + 1]
        model_name = '/'.join(parts[-4:-2])

    model_key = f"{model_name} ({branch})"

    if model_key in model_history:
        vocab_info = model_history[model_key]
        return vocab_info, model_name, branch
    
    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=branch, trust_remote_code=True, use_fast=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=branch, trust_remote_code=True, use_fast=False)

        test_input = [
            {"role": "user", "content": "{UserContent}"},
            {"role": "assistant", "content": "{AIContent}"}
        ]

        vocab_family = get_vocab_family(tokenizer)
        vocab_info = {
            "vocab_family": vocab_family,
            "tokenizer_sha": get_tokenizer_sha(tokenizer),
            "bos": tokenizer.bos_token, 
            "eos": tokenizer.eos_token,
            "pad": tokenizer.pad_token, 
            "unk": tokenizer.unk_token,
            "bos_id": tokenizer.bos_token_id, 
            "eos_id": tokenizer.eos_token_id,
            "pad_id": tokenizer.pad_token_id, 
            "unk_id": tokenizer.unk_token_id,
            "vocab_size": tokenizer.vocab_size,
            "full_vocab_size": tokenizer.vocab_size + tokenizer.added_tokens_encoder.keys().__len__(),
            "default_prompt_format": tokenizer.chat_template == None,
            "prompt_format": tokenizer.apply_chat_template(test_input, tokenize=False, add_generation_prompt=True),
            "tokenizer_class": tokenizer.__class__.__name__ if hasattr(tokenizer, "__class__") else "Unknown"
        }

        model_history[model_key] = vocab_info
        save_history()
    except Exception as e:
        vocab_info = {"error": str(e)}
    
    return vocab_info, model_name, branch

def find_model_info(model_input):
    requests = model_input.split("\n")
    results = []
    for request in requests:
        
        vocab_info, model_name, branch = get_vocab_info(request)

        if vocab_info is None:
            continue

        result_str = f"{model_name} ({branch}): {vocab_info.get('vocab_family', 'Unknown')}"

        if "error" in vocab_info:
            result_str += f', error: {vocab_info["error"]}'
            results.append(result_str)
            continue
        elif vocab_info.get('vocab_family', 'Unknown') == "Unknown":
            result_str += f'\nTokenizer SHA: {vocab_info["tokenizer_sha"]}'
            result_str += f"\nLikely vocab family: {vocab_info['tokenizer_class']}"

        result_str += f"\nBOS: {vocab_info['bos']} id: {vocab_info['bos_id']}"
        result_str += f"\nEOS: {vocab_info['eos']} id: {vocab_info['eos_id']}"
        result_str += f"\nPAD: {vocab_info['pad']} id: {vocab_info['pad_id']}"
        result_str += f"\nUNK: {vocab_info['unk']} id: {vocab_info['unk_id']}"
        result_str += f"\nBase Vocab Size: {vocab_info['vocab_size']}, Full Vocab Size: {vocab_info['full_vocab_size']}"
        prompt_format = vocab_info.get('prompt_format', None)
        if vocab_info['default_prompt_format']:
            result_str += f"\n\n(Likely incorrect) Prompt Format:\n{prompt_format if prompt_format else 'Unknown'}"
        else:
            result_str += f"\n\nPrompt Format:\n{prompt_format if prompt_format else 'Unknown'}"
        
        results.append(result_str)

    history_display = update_history_display()
    return ('\n' + '-' * 130 + '\n').join(results), history_display

with gr.Blocks() as demo:
    with gr.Row(variant="panel"):
        model_input = gr.Textbox(label="Model Name", show_label=True, placeholder="author/model_name", info="Accepts author/model, and full URLs to model repos, e.g. huggingface.co/author/model")
        submit_button = gr.Button(value="Retrieve Model Info", scale=0.955)
    with gr.Row():
        model_info_output = gr.Textbox(label="Model Information", interactive=False)
        model_history_output = gr.Textbox(label="Model History", interactive=False, value=update_history_display())

    submit_button.click(
        fn=find_model_info, 
        inputs=model_input, 
        outputs=[model_info_output, model_history_output]
    )

demo.launch(debug=True)
