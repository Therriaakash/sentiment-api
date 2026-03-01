import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)


class SentimentResponse(BaseModel):
    sentiment: str
    rating: int


@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Analyze the sentiment of this comment: {request.comment}",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_analysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"],
                            },
                            "rating": {"type": "integer", "minimum": 1, "maximum": 5},
                        },
                        "required": ["sentiment", "rating"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        return response.output_parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
