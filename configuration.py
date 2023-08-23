from pydantic_settings import BaseSettings

class Configuration(BaseSettings):
    """
    Settings for the application
    """
    models_dir: str = "./models"
    models_name: str = "bigcode/starcodebase-1b"
    tokenizer_name: str = "bigcode/starcodebase-1b"
    port: int = 5000

    device: str = "cuda"
    EOD: str = "<|endoftext|>"
    FIM_MIDDLE: str = "<fim_middle>"
    FIM_PAD: str = "<fim_pad>"
    FIM_PREFIX: str = "<fim_prefix>"
    FIM_SUFFIX: str = "<fim_suffix>"
