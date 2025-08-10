
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI()

MODEL_NAME = "HoyaHoya/chat-profanity-filter"
try:
    print(f"Loading model from HuggingFace Hub: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model from {MODEL_NAME}: {e}")
    print("Please check if the model exists on HuggingFace Hub.")
    tokenizer, model = None, None

class TextInput(BaseModel):
    text: str

@app.post("/")
async def filter_text(item: TextInput):
    if not model or not tokenizer:
        return {"error": "Model is not loaded. Please train the model first."}

    # 입력 텍스트 토큰화
    inputs = tokenizer(item.text, return_tensors="pt", truncation=True, max_length=512)

    # 모델 예측
    with torch.no_grad():
        logits = model(**inputs).logits

    # 결과 처리
    probabilities = F.softmax(logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item()
    
    is_profanity = prediction == 1  # 1이 hateful(욕설)
    
    result = {
        "text": item.text,
        "is_profanity": is_profanity,
        "confidence": round(confidence, 4)
    }
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
