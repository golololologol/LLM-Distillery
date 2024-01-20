import gradio as gr
from transformers import AutoTokenizer
from dataset_utils import get_vocab_family

model_history = {}
try:
    with open("model_history.txt", "r") as f:
        for line in f:
            model_name, vocab_family = line.strip().split(": ")
            model_history[model_name] = vocab_family
except FileNotFoundError:
    model_history = {}

def find_vocab_family(model_input):
    global model_history

    if "huggingface.co" in model_input:
        model_name = "/".join(model_input.split('/')[-2:])
    else:
        model_name = model_input

    if model_name in model_history:
        history_text = "\n".join([f"{model}: {vocab_family}" for model, vocab_family in model_history.items()])
        return model_history[model_name], history_text
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    message = get_vocab_family(tokenizer)
    if message == "Unknown":
        token = tokenizer.convert_ids_to_tokens(29999)
        message = f"Unknown. Token at id 29999 is `{token}`."

    model_history = {model_name: message, **model_history}

    history_text = "\n".join([f"{model}: {vocab_family}" for model, vocab_family in model_history.items()])

    with open("model_history.txt", "w") as f:
        for model, vocab_family in model_history.items():
            f.write(f"{model}: {vocab_family}\n")
    
    return message, history_text

with gr.Blocks() as demo:
    with gr.Row():
        model_input = gr.Textbox(label="Model Name")
        submit_button = gr.Button(value="Find Vocab Family")
    with gr.Row():
        vocab_family_output = gr.Textbox(label="Vocab Family", interactive=False)
        history_box = gr.Textbox(label="Model History", interactive=False)
    with gr.Column(scale=4):
        submit_button.click(
            fn=find_vocab_family, 
            inputs=model_input, 
            outputs=[vocab_family_output, history_box]
        )

demo.launch(debug=True)
