import os
import json

def save_dataset_and_metadata(dataset_tokenized: list, dataset_content_ranges: list, metadata, save_folder: str):
    file = os.path.join(save_folder, "dataset_tokenized.jsonl")
    metadata_file = os.path.join(save_folder, "dataset_metadata.json")

    with open(file, 'w', encoding='utf-8') as f, open(metadata_file, 'w', encoding='utf-8') as meta_f:
        for i, convo_tokenized in enumerate(dataset_tokenized):
            content_tokens = []
            for content_range in dataset_content_ranges[i]:
                content_start, content_end = content_range
                content_tokens.extend(convo_tokenized[content_start:content_end].tolist())

            data_to_save = {
                "convo_tokenized": convo_tokenized.tolist(),
                "content_ranges": dataset_content_ranges[i],
                "content_tokens": content_tokens
            }
            f.write(json.dumps(data_to_save, ensure_ascii=False) + '\n')

        json.dump(metadata, meta_f, ensure_ascii=False, indent=4)

def save_tokenized_dataset(dataset_tokenized: list, dataset_content_ranges: list, save_folder: str):
    file = os.path.join(save_folder, "dataset_tokenized.jsonl")

    with open(file, 'w', encoding='utf-8') as f:
        for i, convo_tokenized in enumerate(dataset_tokenized):
            content_tokens = []
            for content_range in dataset_content_ranges[i]:
                content_start, content_end = content_range
                content_tokens.extend(convo_tokenized[content_start:content_end].tolist())

            data_to_save = {
                "convo_tokenized": convo_tokenized.tolist(),
                "content_ranges": dataset_content_ranges[i],
                "content_tokens": content_tokens
            }
            f.write(json.dumps(data_to_save, ensure_ascii=False) + '\n')

def load_metadata(distributions_path):
    metadata_path = os.path.join(distributions_path, "dataset_metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as file:
        metadata = json.load(file)
        return metadata
