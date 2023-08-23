import uvicorn
import torch
from fastapi import FastAPI
from pydantic import BaseModel, model_validator, Field
from typing import List, Optional

from model import get_model

from constants import (
    EOD,
    FIM_MIDDLE,
    FIM_PREFIX,
    FIM_SUFFIX,
    device,
    PORT
)

model, tokenizer = get_model()

app = FastAPI()

# parameters = {"max_new_tokens":60,"temperature":0.2,"do_sample":true,"top_p":0.95,"stop":["<|endoftext|>"]}) 

class HFPayload(BaseModel):
    inputs: str
    parameters: dict = Field(exclude=True)
    max_new_tokens: Optional[int] = None
    temperature:  Optional[float] = None
    do_sample:  Optional[bool] = True
    top_p:  Optional[float] = 0.95
    stop:  Optional[List[str]] = None

    @model_validator(mode="after")
    def unpack_parameters(self):
        self.max_new_tokens = self.parameters["max_new_tokens"]
        self.temperature = self.parameters["temperature"]
        self.do_sample = self.parameters["do_sample"]
        self.top_p = self.parameters["top_p"]
        self.stop = self.parameters["stop"]

        return self
    
class CompletionResponse(BaseModel):
    generated_text: str


def codegen(payload: HFPayload) -> str:
    inputs = tokenizer(
        payload.inputs, return_tensors="pt", padding=True, return_token_type_ids=False
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=payload.do_sample,
            temperature=payload.temperature,
            max_new_tokens=payload.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    start = decoded.find(FIM_MIDDLE) + len(FIM_MIDDLE)
    end = decoded.find(EOD, start) or len(decoded)
    completion = decoded[start:end]
    print(f"Generated: {completion}")

    return completion


@app.post("/v1/engines/codegen/completions", response_model=CompletionResponse)
async def completions(payload: HFPayload):
    return CompletionResponse(generated_text=codegen(payload))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=PORT)
