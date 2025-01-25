import json
import random
from tqdm import tqdm
import os
from joblib import Parallel, delayed

def process_sample(sample, rewriting_function, dataset_name):
    return rewriting_function(sample, dataset_name)

def rewrite_dataset(input_path, output_path, rewriting_function, dataset_name, sort):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        lines = infile.readlines()
        samples = Parallel(n_jobs=-1, prefer='threads', pre_dispatch='all')(
            delayed(process_sample)(json.loads(line.strip()), rewriting_function, dataset_name)
            for line in tqdm(lines, desc="Processing samples", unit="samples")
        )
        
        write_records(samples, outfile, sort)

def write_records(samples, outfile, sort):
    if sort:
        print("Sorting samples by length of conversation...")
        samples.sort(key=lambda x: sum([len(turn["value"]) for turn in x['conversations']]))

    for sample in tqdm(samples, desc="Writing samples", unit="samples"):
        outfile.write(json.dumps(sample, ensure_ascii=False) + '\n')



def rewrite_alpaca_gpt4(sample, dataset_name):
    instruction = sample['instruction']
    input_text = sample['input']
    output_text = sample['output']

    # 50/50 chance for space or newline between instruction and input
    separator = '\n' if random.random() < 0.5 else ' '
    combined_instruction = instruction if not input_text else f"{instruction}{separator}{input_text}"

    new_sample = {
        "init": "",
        "conversations": [{"from": "human", "value": combined_instruction}, {"from": "gpt", "value": output_text}],
        "source": dataset_name,
        "tags": []
    }

    return new_sample

def rewrite_codefeedback(sample, dataset_name):
    instruction = sample['query']
    response = sample['answer']

    new_sample = {
        "init": "",
        "conversations": [{"from": "human", "value": instruction}, {"from": "gpt", "value": response}],
        "source": dataset_name,
        "tags": []
    }

    return new_sample

def rewrite_multitask(sample, dataset_name):
    init = sample['system']
    instruction = sample['prompt']
    output_text = sample['chosen']

    new_sample = {
        "init": init,
        "conversations": [{"from": "human", "value": instruction}, {"from": "gpt", "value": output_text}],
        "source": dataset_name,
        "tags": []
    }

    return new_sample

def rewrite_dolly(sample, dataset_name):
    instruction = sample['instruction'].strip()
    input_text = sample['context'].strip()
    output_text = sample['response'].strip()

    # a random chance to pick any separator from the list
    separator = random.choice(['\n', ' ', '  ', '  ', '\n\n', '\n\n\n', ' | '])
    combined_instruction = instruction if not input_text else f"{input_text}{separator}{instruction}"

    new_sample = {
        "init": "",
        "conversations": [{"from": "human", "value": combined_instruction}, {"from": "gpt", "value": output_text}],
        "source": dataset_name,
        "tags": []
    }

    return new_sample

def rewrite_gpt_roleplay(sample, dataset_name):
    dialogues = sample.get('dialogues', [])
    sys = sample.get('context', "")
    greeting = sample.get('greeting', "")

    if not dialogues:
        return None
    
    samples = []
    for chat in dialogues:
        converted_chat = []

        if greeting:
            converted_chat.append({"from": "gpt", "value": greeting})

        for turn in chat['chat']:
            speaker = "gpt" if turn['role'] == 'char' else "human"
            converted_chat.append({"from": speaker, "value": turn['content']})

        new_sample = {
            "init": sys if sys else "",
            "conversations": converted_chat,
            "source": dataset_name,
            "tags": []
        }

        samples.append(new_sample)

    return samples

def rewrite_random_samples(sample, dataset_name):
    text = sample['text']

    new_sample = {
        "init": "",
        "conversations": [{"from": "human", "value": text}],
        "source": dataset_name,
        "tags": ["completion"]
    }

    return new_sample

def rewrite_sharegpt(sample, dataset_name):
    convo = sample["conversations"]
    sys = ""
    turns = []
    for turn in convo:
        if turn["from"] == "system":
            sys = turn["value"]
        else:
            turns.append({"from": turn["from"], "value": turn["value"]})

    # drop the system message with 75% chance
    if random.random() < 0.75:
        sys = ""

    new_sample = {
        "init": sys,
        "conversations": turns,
        "source": dataset_name,
        "tags": []
    }

    return new_sample

def rewrite_magpie_reasoning(sample, dataset_name):
    instruction = sample['instruction']
    output_text = sample['response']

    new_sample = {
        "init": "",
        "conversations": [{"from": "human", "value": instruction}, {"from": "gpt", "value": output_text}],
        "source": dataset_name,
        "tags": []
    }

    return new_sample

def rewrite_custom(sample, dataset_name): # used to update from an older format pipeline used
    sys = sample.get('init', "")
    source = sample.get('source', dataset_name)
    tags = sample.get('tags', [])
    reversed = "reversed" in tags or sample.get('reversed', False)
    turns = []

    for i, turn_text in enumerate(sample['conversations']):

        if reversed:
            assistant = (i + 1) % 2
        else:
            assistant = i % 2

        new_turn = {
            "from": "gpt" if assistant else "human",
            "value": turn_text
        }

        turns.append(new_turn)

    new_sample = {
        "init": sys,
        "conversations": turns,
        "source": source,
        "tags": tags
    }

    return new_sample


input_file_path = r"C:\Users\User\Downloads\train-00008-of-00009.jsonl"
rewriting_function = rewrite_magpie_reasoning # change this line to the appropriate function

path = os.path.dirname(input_file_path)
name = os.path.basename(input_file_path).split('.')[0]
output_file_path = os.path.join(path, f"Converted_{name}.jsonl")
dataset_name = name
sort = True

rewrite_dataset(input_file_path, output_file_path, rewriting_function, dataset_name, sort)
print(f"Converted {input_file_path} to {output_file_path} with dataset name {dataset_name}")
