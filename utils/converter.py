import json
import random

def write_records(records, outfile, sort):
    if sort:
        sorted_records = sorted(records, key=lambda x: len(x['init']))

        for record in sorted_records:
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

    else:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

def rewrite_alpaca_gpt4(input_path, output_path, dataset_name, sort):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        records = []
        for line in infile:
            record = json.loads(line.strip())
            instruction = record['instruction']
            input_text = record['input']
            output_text = record['output']

            # 50/50 chance for space or newline between instruction and input
            separator = '\n' if random.random() < 0.5 else ' '
            combined_instruction = instruction if not input_text else f"{instruction}{separator}{input_text}"

            new_record = {
                "init": "",
                "conversations": [combined_instruction, output_text],
                "source": dataset_name,
                "tags": []
            }

            records.append(new_record)

        write_records(records, outfile, sort)


def rewrite_codefeedback(input_path, output_path, dataset_name, sort):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        records = []
        for line in infile:
            record = json.loads(line.strip())
            instruction = record['query']
            response = record['answer']

            new_record = {
                "init": "",
                "conversations": [instruction, response],
                "source": dataset_name,
                "tags": []
            }

            records.append(new_record)

        write_records(records, outfile, sort)


def rewrite_multitask(input_path, output_path, dataset_name, sort):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        records = []
        for line in infile:
            record = json.loads(line.strip())
            init = record['system']
            instruction = record['prompt']
            output_text = record['chosen']

            new_record = {
                "init": init,
                "conversations": [instruction,output_text],
                "source": dataset_name,
                "tags": []
            }

            records.append(new_record)

        write_records(records, outfile, sort)

def convert_dolly(input_path, output_path, dataset_name, sort):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        records = []
        for line in infile:
            record = json.loads(line.strip())
            instruction = record['instruction'].strip()
            input_text = record['context'].strip()
            output_text = record['response'].strip()

            # a random chance to pick any separator from the list
            separator = random.choice(['\n', ' ', '  ', '  ', '\n\n', '\n\n\n', ' | '])
            combined_instruction = instruction if not input_text else f"{input_text}{separator}{instruction}"

            new_record = {
                "init": "",
                "conversations": [combined_instruction, output_text],
                "source": dataset_name,
                "tags": []
            }

            records.append(new_record)

        write_records(records, outfile, sort)


def convert_gpt_roleplay(input_path, output_path, dataset_name, sort):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        records = []
        for line in infile:
            record = json.loads(line.strip())
            dialogues = record.get('dialogues', [])
            sys = record.get('context', "")
            greeting = record.get('greeting', "")

            if not dialogues:
                continue
            
            for chat in dialogues:
                converted_chat = []

                if greeting:
                    converted_chat.append(greeting)

                for turn in chat['chat']:
                    converted_chat.append(turn["content"])

                new_record = {
                    "init": sys if sys else "",
                    "conversations": converted_chat,
                    "source": dataset_name,
                    "tags": ["reversed"] if greeting else []
                }

                records.append(new_record)

        write_records(records, outfile, sort)


input_file_path = r"C:\Users\PC\Downloads\en-00000-of-00001-6291ef0dc79c47ed.jsonl"
output_file_path = r"C:\Users\PC\Downloads\Converted_gpt_roleplay.jsonl"
dataset_name = 'gpt_roleplay'
sort = True

convert_gpt_roleplay(input_file_path, output_file_path, dataset_name, sort)
print(f"Converted {input_file_path} to {output_file_path} with dataset name {dataset_name}")
