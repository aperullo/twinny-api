from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


from constants import (
    EOD,
    FIM_MIDDLE,
    FIM_PREFIX,
    FIM_SUFFIX,
    FIM_PAD,
    tokenizer_name,
    model_name,
)

special_tokens = {
    "additional_special_tokens": [
        EOD,
        FIM_PREFIX,
        FIM_MIDDLE,
        FIM_SUFFIX,
        FIM_PAD,
    ],
    "pad_token": EOD,
}


def get_model():
    tokenizer = AutoTokenizer.from_pretrained("./models/bigcode/starcodebase-1b/", padding_side="left", local_files_only=True)
    tokenizer.add_special_tokens(special_tokens)
    model = AutoModelForCausalLM.from_pretrained("./models/bigcode/starcodebase-1b/", device_map="auto", local_files_only=True)

    return model, tokenizer
