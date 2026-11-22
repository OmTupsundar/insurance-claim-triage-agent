import json
import os
from typing import Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()


class ExtractedClaimData(BaseModel):
    claim_id: str | None = Field(default=None, description="Claim ID if mentioned in document")
    policy_id: str | None = Field(default=None, description="Policy ID or policy number if found")
    claimant_name: str | None = Field(default=None, description="Name of policyholder or claimant")
    insurance_type: str | None = Field(default=None, description="Health, Motor, Property, Travel, etc.")
    claimed_amount_inr: float | None = Field(default=None, description="Claimed amount in INR if stated")
    incident_summary: str | None = Field(default=None, description="Short summary of incident or treatment")
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Important claim fields missing or unclear in the document",
    )
    verified_facts: list[str] = Field(
        default_factory=list,
        description="Facts explicitly supported by the extracted text",
    )


def analyze_claim_document(
    document_text: str,
    claim_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Analyze extracted claim PDF text using Gemini and return structured fields
    along with missing information and verified facts.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

    if not api_key:
        return {
            "success": False,
            "error": "GEMINI_API_KEY is missing in .env.",
            "data": None,
        }

    if not document_text or not document_text.strip():
        return {
            "success": False,
            "error": "No document text provided for analysis.",
            "data": None,
        }

    client = genai.Client(api_key=api_key)

    system_instruction = """
You are the Document Analysis Agent for ClaimPilot AI.
Extract only what is present in the document and user-supplied metadata.
Never guess or invent missing numbers, policy IDs, or dates.
List any missing or unclear details in `missing_fields`.
Put only explicitly stated facts into `verified_facts`.
"""

    prompt = f"""
User Metadata (for reference only):
{json.dumps(claim_metadata or {}, indent=2)}

Extracted PDF Text:
{document_text}
"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=ExtractedClaimData,
                temperature=0.1,
            ),
        )
        parsed = json.loads(response.text)
        return {
            "success": True,
            "error": None,
            "data": parsed,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Document analysis failed: {exc}",
            "data": None,
        }
    