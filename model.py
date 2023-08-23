from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from configuration import Configuration
from pathlib import Path

def get_model(config: Configuration):

    special_tokens = {
    "additional_special_tokens": [
        config.EOD,
        config.FIM_PREFIX,
        config.FIM_MIDDLE,
        config.FIM_SUFFIX,
        config.FIM_PAD,
    ],
    "pad_token": config.EOD,
}

    tokenizer_path = f"{config.models_dir}/{config.tokenizer_name}/"
    model_path = f"{config.models_dir}/{config.models_name}/"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left", local_files_only=True)
    tokenizer.add_special_tokens(special_tokens)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", local_files_only=True)

    return model, tokenizer
