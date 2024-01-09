import json

def compare_vocabs(file1, file2, output_file):
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
    
    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(result, output, indent=4)
    
#compare_vocabs(r"F:\trained\BallCrusher9000\dataset_metadata.json", r"F:\distilled\janny_Filteredtest\neural-chat-7b-v3-1-exl2\dataset_metadata.json", "comparison_result.json")
