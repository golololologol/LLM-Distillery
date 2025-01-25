import pandas as pd
import numpy as np
import base64
import json
import glob
import os

def convert_parquet_to_jsonl(parquet_pattern):
    files = sorted(glob.glob(parquet_pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {parquet_pattern}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    jsonl_path = os.path.splitext(files[-1])[0].replace('-00000-of', '') + '.jsonl'
    
    def handle_item(item):
        if isinstance(item, bytes):
            return base64.b64encode(item).decode('utf-8')
        elif isinstance(item, np.generic):
            return item.item()
        elif isinstance(item, np.ndarray):
            if item.dtype == object:
                return [handle_item(x) for x in item]
            if item.dtype == np.uint8:
                return base64.b64encode(item).decode('utf-8')
            else:
                return item.tolist()
        elif isinstance(item, pd.Timestamp):
            return item.isoformat()
        elif isinstance(item, (dict, pd.Series)):
            return {k: handle_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [handle_item(x) for x in item]
        return item

    with open(jsonl_path, 'w', encoding='utf-8') as file:
        for _, row in df.iterrows():
            corrected_row = {key: handle_item(value) for key, value in row.to_dict().items()}
            json_string = json.dumps(corrected_row, ensure_ascii=False)
            file.write(json_string + '\n')

    with open(jsonl_path, 'r', encoding='utf-8') as file:
        sample = json.loads(file.readline())
        def crop_value(val, max_len=50):
            return val[:max_len] if isinstance(val, str) and len(val) > max_len else val
        dataset_format = "\n ".join([f"{key}: {crop_value(sample[key])}" for key in sample.keys()])
        
    return dataset_format


# replace the name of the parquet file with the pattern, e.g. from 'train-00000-of-00001.parquet' to 'train-*.parquet'
parquet_pattern = r"C:\Users\User\Downloads\train-*.parquet"
dataset_format = convert_parquet_to_jsonl(parquet_pattern)
print("Conversion complete.")
print(f"Dataset format:\n{dataset_format}")