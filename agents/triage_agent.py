import json
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
POLICIES_PATH = BASE_DIR / "data" / "policies.json"


def get_policy_details(policy_id: str) -> str:
    """Fetches coverage rules, sum insured, deductible, and exclusions for a given policy ID."""
    with open(POLICIES_PATH, "r", encoding="utf-8") as f:
        policies = json.load(f)
    policy = policies.get(policy_id.upper())
    if not policy:
        return json.dumps({"error": f"Policy {policy_id} not found"})
    return json.dumps({"policy_id": policy_id.upper(), **policy})


def check_claim_completeness(has_incident_date: bool, has_location: bool, has_damage_summary: bool, has_supporting_proof: bool) -> str:
    """Checks whether required claim details and supporting proof (FIR, photos, or medical/hospital bills) are present."""
    missing = []
    if not has_incident_date:
        missing.append("incident_date")
    if not has_location:
        missing.append("location")
    if not has_damage_summary:
        missing.append("damage_summary")
    if not has_supporting_proof:
        missing.append("supporting_proof (FIR/photos/bills)")
    return json.dumps({
        "is_complete": len(missing) == 0,
        "missing_fields": missing
    })


def calculate_triage_route(estimated_claim_inr: int, fast_track_limit_inr: int, is_complete: bool, possible_exclusion: bool) -> str:
    """Determines whether a claim should go to Fast-Track Auto-Triage, Manual Adjuster Review, or Needs Info."""
    if not is_complete:
        route = "NEEDS_MORE_INFO"
        reason = "Mandatory claim fields or supporting documents are missing."
    elif possible_exclusion:
        route = "MANUAL_SIU_REVIEW"
        reason = "Claim mentions a potential policy exclusion or red flag."
    elif estimated_claim_inr <= fast_track_limit_inr:
        route = "FAST_TRACK"
        reason = f"Claim amount (₹{estimated_claim_inr}) is within fast-track limit (₹{fast_track_limit_inr}) and complete."
    else:
        route = "MANUAL_ADJUSTER_REVIEW"
        reason = f"Claim amount (₹{estimated_claim_inr}) exceeds fast-track limit (₹{fast_track_limit_inr})."
    return json.dumps({"route": route, "reason": reason})


SYSTEM_INSTRUCTION = """
You are ClaimPilot AI, an insurance claims triage and routing agent.
For every incoming claim:
1. Call `get_policy_details` using the provided policy ID.
2. Call `check_claim_completeness` based on what details/proofs are provided.
3. Call `calculate_triage_route` using the policy's fast_track_limit_inr, completeness result, estimated claim amount in INR, and whether any exclusion applies.
4. Provide a clear, structured triage report with: Policy Summary, Coverage & Exclusion Check, Routing Decision, and Next Steps.
"""

TOOLS = [get_policy_details, check_claim_completeness, calculate_triage_route]


def triage_claim(claim_prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=claim_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=TOOLS,
            temperature=0.2,
        ),
    )
    return response.text