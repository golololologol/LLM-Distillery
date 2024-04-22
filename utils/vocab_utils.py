from transformers import AutoTokenizer
import hashlib

def try_load_tokenizer(model_path: str) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    except:
        raise ValueError(f"Tokenizer for {model_path} could not be loaded")
    
    return tokenizer

def get_special_tokens(tokenizer=None, model_path="") -> dict[str, str|int]:
    tokenizer = try_load_tokenizer(model_path) if tokenizer == None else tokenizer

    sp_toks = {
        "bos": tokenizer.bos_token,
        "bos_id": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token,
        "eos_id": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token,
        "pad_id": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token,
        "unk_id": tokenizer.unk_token_id
    }
    
    return sp_toks

def get_tokenizer_sha(tokenizer = None, model_path="") -> str:
    tokenizer = try_load_tokenizer(model_path) if tokenizer == None else tokenizer

    base_tokens = sorted(tokenizer._tokenizer.get_vocab(with_added_tokens=False).keys())
    tokenizer_sha = hashlib.sha256("".join(base_tokens).encode()).hexdigest()
    return tokenizer_sha

def get_vocab_family(tokenizer=None, model_path="") -> str:
    tokenizer = try_load_tokenizer(model_path) if tokenizer == None else tokenizer

    tokenizer_sha = get_tokenizer_sha(tokenizer)

    sha_to_family = {
        "943d2a9918de54ad09e974a6d834ba44191fa98210c1414a822338fcf3f54038": "mistral",
        "d1e4f1ef2bea057b346dd69244790543bec0dca751118e523d62ee6ae30b6036": "llama_1|2",
        "7f1739ca602e925d6c93e42c0d4d4f6f607169f5b016d5d322e4ec42a1c7c563": "llama_3",
        "55b3930bf06e1139760a921273642322f38c2165620b34f1e6d521a7bc431c90": "command-r",
        "24c4ea5aecffe896ab422b984ec1e4598d5aa565417432271ae752594a71e94e": "dbrx",
        "99fe18abd82f35c8ef2a82b1277059dbe16d0a033fecdfa7726a3a6f574a3866": "gpt2",
        "5520ff09e574cbb231f2baf283bd1a2e358b5d747c1a6aaa6402ee03991a409d": "qwen2",
        "21d0e3b8f0c9f89b7b6f9a4c5b72b6ba2f70e7805460ca4cbd4b9c55d29f31fc": "gptneox",
        "20c2a98cfdd0ae39798458b5a6dc4838d1b700a0ec074525654a57bfa3088648": "gemma",
        "4088308e46cd5596c6c320bd102e48ee33b03d42d12447ccce8beafe4310bfa3": "yi",
        "2e7d13c6f9a9825b1dfeb645fe3130b118e4c119bdf0460be06bd7e2d7660728": "deepseek",
        "62947c306f3a11187ba2a4a6ea25de91ce30c5724e6b647a1d6f0f8868217ead": "deepseek_1.5",
        "0d35d803249ab1f34c99f8a10a562fc433134cca4fbd566dcd6ca61c4a857b04": "T5"
    }

    vocab_family = sha_to_family.get(tokenizer_sha, "Unknown") # type: ignore
    return vocab_family