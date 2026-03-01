import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------
# Request Schema
# ---------------------------
class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)


# ---------------------------
# Response Schema
# ---------------------------
class SentimentResponse(BaseModel):
    sentiment: str
    rating: int


# ---------------------------
# POST /comment
# ---------------------------
@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):

    # Validate empty or whitespace comment
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis system. "
                        "Respond ONLY with valid JSON in this format:\n"
                        '{ "sentiment": "positive|negative|neutral", "rating": 1-5 }\n'
                        "Rating scale:\n"
                        "5 = highly positive\n"
                        "4 = positive\n"
                        "3 = neutral\n"
                        "2 = negative\n"
                        "1 = highly negative\n"
                        "Do not include explanations."
                    ),
                },
                {"role": "user", "content": request.comment},
            ],
            response_format={"type": "json_object"},
        )

        # Extract text output safely
        result_text = response.output_text

        # Convert JSON string to dictionary
        result_json = json.loads(result_text)

        return result_json

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON returned by model")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
