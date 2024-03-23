from transformers import AutoTokenizer

def get_special_tokens(vocab_family) -> dict[str, str|int]:
    if (vocab_family == "llama") or (vocab_family == "mistral"):
        sp_toks = {
            "bos": "<s>",
            "bos_id": 1,
            "eos": "</s>",
            "eos_id": 2,
            "pad": None,
            "pad_id": None,
            "unk": "<unk>",
            "unk_id": 0
        }
    else:
        raise NotImplementedError(f"{vocab_family} not yet supported")
    return sp_toks

def get_vocab_family(tokenizer=None, model_path="") -> str:
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

    token_29999_to_family = {
        "监": "mistral",
        "Z": "llama"
    }

    token_29999 = tokenizer.convert_ids_to_tokens(29999)
    vocab_family = token_29999_to_family.get(token_29999, "Unknown") # type: ignore
    return vocab_family