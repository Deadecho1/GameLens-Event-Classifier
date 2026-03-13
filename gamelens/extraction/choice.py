import base64
import json
import os

# Assuming this exists in your project
from fastapi import APIRouter, File, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_PROMPT_CHOICE = """
        You are analyzing a roguelike game screenshot.
        Extract the item/upgrade titles shown and identify the currently selected one.
        """

# Initialize the router and OpenAI client
router = APIRouter(prefix="/api/v1/choice", tags=["choice"])
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else OpenAI()


class ExtractionResponse(BaseModel):
    choices: list[str]
    selected_choice: str


@router.post("/extract-choices", response_model=ExtractionResponse)
async def extract_choices(
    file: UploadFile = File(...),
    prompt: str = DEFAULT_PROMPT_CHOICE,
    model: str = "gpt-5.1",
):
    """
    Analyzes a roguelike game screenshot to extract choice titles
    and identify the currently selected one.
    """
    # Validate the uploaded file is an image
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(
            status_code=400, detail="Could not read the uploaded image."
        )

    b64_encoded = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{file.content_type};base64,{b64_encoded}"

    schema = {
        "name": "choice_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Choice titles, in on-screen order.",
                },
                "selected_choice": {
                    "type": "string",
                    "description": "The currently selected choice title, or empty string if none.",
                },
            },
            "required": ["choices", "selected_choice"],
            "additionalProperties": False,
        },
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_schema", "json_schema": schema},
            messages=[
                {
                    "role": "system",
                    "content": (prompt),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Return all choice titles and the selected choice.",
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )

        content = resp.choices[0].message.content
        return json.loads(content) if isinstance(content, str) else content

    except Exception as e:
        # Catch and surface OpenAI API errors gracefully
        raise HTTPException(
            status_code=500, detail=f"Failed to process image with OpenAI: {str(e)}"
        )
