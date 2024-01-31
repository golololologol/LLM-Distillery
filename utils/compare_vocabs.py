import json
import os

def compare_vocabs(folder1, folder2):
    file1 = os.path.join(folder1, "dataset_metadata.json")
    file2 = os.path.join(folder2, "dataset_metadata.json")
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        metadata1 = json.load(f1)
        metadata2 = json.load(f2)
    
    vocab1 = metadata1.get("vocab", {})
    vocab2 = metadata2.get("vocab", {})
    
    common_tokens = set(vocab1.keys()) & set(vocab2.keys())
    unique_tokens_dict1 = set(vocab1.keys()) - common_tokens
    unique_tokens_dict2 = set(vocab2.keys()) - common_tokens
    
    num_common_tokens = len(common_tokens)
    
    result = {
        "Number of identical tokens": num_common_tokens,
        "Keys unique to dict1": list(unique_tokens_dict1),
        "Keys unique to dict2": list(unique_tokens_dict2),
        "Number of unique tokens in dict1": len(unique_tokens_dict1),
        "Number of unique tokens in dict2": len(unique_tokens_dict2)
    }
    print(result)
    
compare_vocabs(r"F:\distilled\data-MNHTN-standardized-Puffin\Nous-Hermes-Llama2-13b", r"F:\distilled\data-MNHTN-standardized-Puffin\UtopiaXL-13B")
