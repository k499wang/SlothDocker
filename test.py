from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import TextStreamer
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch

# Initialize the FastAPI app
app = FastAPI()

# Load the model and tokenizer (this will be done once on server startup)
MODEL_NAME = "k3vinwvng/NLPDocker"
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Enable faster inference
FastLanguageModel.for_inference(model)

# Define input data structure
class InferenceRequest(BaseModel):
    input_text: str

# Set up a TextStreamer to handle output formatting
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

# Define an endpoint for inference
@app.post("/generate/")
async def generate_response(request: InferenceRequest):
    try:
        # Prepare input text
        input_text = f"""Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
Translate This into a Docker Command: {request.input_text}

### Response:
"""

        # Tokenize input
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda" if torch.cuda.is_available() else "cpu")

        # Generate output using the model
        output = model.generate(input_ids, streamer=text_streamer, max_new_tokens=200, use_cache=True, top_p=0.9,
                                no_repeat_ngram_size=2, num_return_sequences=1)

        # Return the result
        return {"response": output}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with: uvicorn your_file_name:app --reload
