import json
import random
import os

def write_records(samples, outfile, sort):
    if sort:
        samples.sort(key=lambda x: sum([len(turn) for turn in x['conversations']]))

    for sample in samples:
        outfile.write(json.dumps(sample, ensure_ascii=False) + '\n')

def rewrite_alpaca_gpt4(input_path, output_path, dataset_name, sort):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        samples = []
        for line in infile:
            sample = json.loads(line.strip())
            instruction = sample['instruction']
            input_text = sample['input']
            output_text = sample['output']

            # 50/50 chance for space or newline between instruction and input
            separator = '\n' if random.random() < 0.5 else ' '
            combined_instruction = instruction if not input_text else f"{instruction}{separator}{input_text}"

            new_sample = {
                "init": "",
                "conversations": [combined_instruction, output_text],
                "source": dataset_name,
                "tags": []
            }

            samples.append(new_sample)

        write_records(samples, outfile, sort)


def rewrite_codefeedback(input_path, output_path, dataset_name, sort):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        samples = []
        for line in infile:
            sample = json.loads(line.strip())
            instruction = sample['query']
            response = sample['answer']

            new_sample = {
                "init": "",
                "conversations": [instruction, response],
                "source": dataset_name,
                "tags": []
            }

            samples.append(new_sample)

        write_records(samples, outfile, sort)


def rewrite_multitask(input_path, output_path, dataset_name, sort):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        samples = []
        for line in infile:
            sample = json.loads(line.strip())
            init = sample['system']
            instruction = sample['prompt']
            output_text = sample['chosen']

            new_sample = {
                "init": init,
                "conversations": [instruction,output_text],
                "source": dataset_name,
                "tags": []
            }

            samples.append(new_sample)

        write_records(samples, outfile, sort)

def convert_dolly(input_path, output_path, dataset_name, sort):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        samples = []
        for line in infile:
            sample = json.loads(line.strip())
            instruction = sample['instruction'].strip()
            input_text = sample['context'].strip()
            output_text = sample['response'].strip()

            # a random chance to pick any separator from the list
            separator = random.choice(['\n', ' ', '  ', '  ', '\n\n', '\n\n\n', ' | '])
            combined_instruction = instruction if not input_text else f"{input_text}{separator}{instruction}"

            new_sample = {
                "init": "",
                "conversations": [combined_instruction, output_text],
                "source": dataset_name,
                "tags": []
            }

            samples.append(new_sample)

        write_records(samples, outfile, sort)


def convert_gpt_roleplay(input_path, output_path, dataset_name, sort):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        samples = []
        for line in infile:
            sample = json.loads(line.strip())
            dialogues = sample.get('dialogues', [])
            sys = sample.get('context', "")
            greeting = sample.get('greeting', "")

            if not dialogues:
                continue
            
            for chat in dialogues:
                converted_chat = []

                if greeting:
                    converted_chat.append(greeting)

                for turn in chat['chat']:
                    converted_chat.append(turn["content"])

                new_sample = {
                    "init": sys if sys else "",
                    "conversations": converted_chat,
                    "source": dataset_name,
                    "tags": ["reversed"] if greeting else []
                }

                samples.append(new_sample)

        write_records(samples, outfile, sort)

def rewrite_random_smaples(input_path, output_path, dataset_name, sort):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        samples = []
        for line in infile:
            sample = json.loads(line.strip())

            text = sample['text']

            new_sample = {
                "init": "",
                "conversations": [text] if text else [""],
                "source": dataset_name,
                "tags": ["completion"]
            }

            samples.append(new_sample)

        write_records(samples, outfile, sort)

def rewrite_sharegpt(input_path, output_path, dataset_name, sort):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        samples = []
        for line in infile:
            convo = json.loads(line.strip())["conversations"]
            sys = ""
            turns = []
            for turn in convo:
                if turn["from"] == "system":
                    sys = turn["value"]
                else:
                    turns.append(turn["value"])

            # drop the system message with 75% chance
            if random.random() < 0.75:
                sys = ""

            new_sample = {
                "init": sys,
                "conversations": turns,
                "source": dataset_name,
                "tags": []
            }

            samples.append(new_sample)

        write_records(samples, outfile, sort)


input_file_path = r"C:\Users\PC\Downloads\opus-writing-prompts-2-sharegpt.jsonl"
path = os.path.dirname(input_file_path)
name = os.path.basename(input_file_path).split('.')[0]
output_file_path = os.path.join(path, f"Converted_{name}.jsonl")
dataset_name = name
sort = True

rewrite_sharegpt(input_file_path, output_file_path, dataset_name, sort)
print(f"Converted {input_file_path} to {output_file_path} with dataset name {dataset_name}")
