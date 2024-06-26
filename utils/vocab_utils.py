from transformers import AutoTokenizer
import hashlib


def try_load_tokenizer(model_path: str) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    except:
        raise ValueError(f"Tokenizer for {model_path} could not be loaded")
    
    return tokenizer


def get_tokenizer_sha(tokenizer = None, model_path="") -> str:
    tokenizer = try_load_tokenizer(model_path) if tokenizer == None else tokenizer

    all_tokens = tokenizer.get_vocab().keys()
    added_tokens = tokenizer.get_added_vocab().keys()
    base_tokens = sorted(set(all_tokens) - set(added_tokens))
    tokenizer_sha = hashlib.sha256("".join(base_tokens).encode()).hexdigest()
    return tokenizer_sha


def get_family_bos_id(vocab_family: str, tokenizer=None, model_path="") -> int|None:
    family_to_bos_id = {
        "mistral": 1,
        "llama_1|2": 1,
        "llama_3": 128000,
        "command-r": 5,
        "dbrx": None,
        "gpt2": 50256,
        "codeqwen": 1,
        "qwen_1.5": None,
        "gptneox": 50256,
        "gemma": 2,
        "yi": 1,
        "deepseek": None,
        "deepseek_1.5": 100000,
        "T5": None,
        "codellama": 1,
        "jamba": 1
    }

    bos_id = family_to_bos_id.get(vocab_family, None)

    if bos_id is None:
        tokenizer = try_load_tokenizer(model_path) if tokenizer == None else tokenizer
        bos_id = tokenizer.bos_token_id
        
    return bos_id


def get_vocab_family(tokenizer=None, model_path="") -> str:
    tokenizer = try_load_tokenizer(model_path) if tokenizer == None else tokenizer

    tokenizer_sha = get_tokenizer_sha(tokenizer)

    sha_to_family = {
        "154a07d332d0466ce54d5e83190930dc872c95777c493653c48d6b6b01891377": "mistral",
        "88dfafd1e6cd6fc3cf71600f1c8590ec6b457263267d801636320000a6f687e3": "llama_1|2",
        "7f1739ca602e925d6c93e42c0d4d4f6f607169f5b016d5d322e4ec42a1c7c563": "llama_3",
        "748d7e81288e3b759c4458794fee17f546a07e08e70832d2f04486cd1fc76121": "command-r",
        "12c2a843415f76681408ba2645bb7ea68fc00e864afbfa408193159f847fbe4b": "dbrx",
        "f41d538d54aa627ad2d1a7d853759ac104b94775a5de51e6e6e6e112fb32c1de": "gpt2",
        "e59f183b9781a484efd580deda830bd952f7d3694c4648d41d295284a48f2945": "codeqwen",
        "5520ff09e574cbb231f2baf283bd1a2e358b5d747c1a6aaa6402ee03991a409d": "qwen_1.5",
        "347b0481e0536ab58d630b461c8bbb9dceb8b1a3ff6b9250c73bcd423e64fa71": "gptneox",
        "9eb38d83274ea4aac4d1a037331c7b9f51ee5a74bc2ff9d85f3c1b7a944f2fd0": "gemma",
        "b6f82ad160f599b1dd5bec8987acb5a316423d04de3123fa1eb0c8f1ba7f5568": "gemma",
        "f6556674148d92703237bab474c2cf220255926e7b6811e526b072d0ed086beb": "yi",
        "2e7d13c6f9a9825b1dfeb645fe3130b118e4c119bdf0460be06bd7e2d7660728": "deepseek",
        "62947c306f3a11187ba2a4a6ea25de91ce30c5724e6b647a1d6f0f8868217ead": "deepseek_1.5",
        "94c18f1464d6aeb4542dff2fb8dc837e131e39853d86707eea683470c7344480": "T5",
        "cabd41803ba4aa362c59603aa9fedd80d8eab202708beccce9f4e1e0b58eaf3f": "codellama",
        "c2ed819dc3c535a3a64a10d492a39baa87b9cc7aa0a2c72adecc1b31e3e1b544": "jamba"
    }

    vocab_family = sha_to_family.get(tokenizer_sha, "Unknown") # type: ignore
    return vocab_family