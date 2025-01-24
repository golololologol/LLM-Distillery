import pandas as pd
import numpy as np
import base64
import json


def convert_parquet_to_jsonl(parquet_path):
    jsonl_path = parquet_path.replace('.parquet', '.jsonl')
    df = pd.read_parquet(parquet_path)
    
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

parquet_file_path = r"C:\Users\PC\Downloads\en-00000-of-00001-6291ef0dc79c47ed.parquet"
convert_parquet_to_jsonl(parquet_file_path)
print('Conversion complete.')
