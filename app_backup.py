# import os
# import time
# import fitz  # pymupdf
# import pandas as pd
# import streamlit as st
# from dotenv import load_dotenv
# from google import genai
# from google.genai import types

# load_dotenv()

# st.set_page_config(page_title="ClaimPilot-AI | Insurance Agent", page_icon="🛡️", layout="wide")
# st.title("🛡️ ClaimPilot-AI: Autonomous Insurance Claim Triage Agent")
# st.caption("Upload a claim/medical/repair PDF or enter claim details. The AI agent autonomously verifies policy status, checks exclusions, and recommends adjudication.")

# api_key = os.getenv("GEMINI_API_KEY")
# if not api_key or api_key == "your_actual_api_key_here":
#     st.error("Please add your valid GEMINI_API_KEY inside the .env file and refresh.")
#     st.stop()

# client = genai.Client(api_key=api_key)

# # --- 1. Mock Policy & Claim Database (Pandas) ---
# POLICIES_DB = pd.DataFrame([
#     {
#         "policy_id": "POL-1001",
#         "holder_name": "Rohan Deshmukh",
#         "policy_type": "Comprehensive Health",
#         "status": "Active",
#         "coverage_limit_inr": 500000,
#         "deductible_inr": 5000,
#         "waiting_period_completed": True,
#         "exclusions": "Cosmetic surgery, experimental treatments, pre-existing conditions within 24 months"
#     },
#     {
#         "policy_id": "POL-1002",
#         "holder_name": "Sneha Kulkarni",
#         "policy_type": "Zero-Dep Motor Comprehensive",
#         "status": "Active",
#         "coverage_limit_inr": 800000,
#         "deductible_inr": 2000,
#         "waiting_period_completed": True,
#         "exclusions": "Drunk driving, driving without valid license, consequential engine water damage"
#     },
#     {
#         "policy_id": "POL-1003",
#         "holder_name": "Amit Patil",
#         "policy_type": "Comprehensive Health",
#         "status": "Lapsed",
#         "coverage_limit_inr": 300000,
#         "deductible_inr": 5000,
#         "waiting_period_completed": False,
#         "exclusions": "All claims invalid while policy status is Lapsed"
#     }
# ])

# # --- 2. Agent Tools (Python functions Gemini can call autonomously) ---
# def lookup_policy(policy_id: str) -> dict:
#     """Fetches policy details, coverage limit, deductible, status, and exclusions by Policy ID."""
#     match = POLICIES_DB[POLICIES_DB["policy_id"].str.upper() == policy_id.strip().upper()]
#     if match.empty:
#         return {"found": False, "message": f"No policy found for ID {policy_id}."}
#     return {"found": True, "policy": match.iloc[0].to_dict()}

# def calculate_payout(claimed_amount_inr: float, coverage_limit_inr: float, deductible_inr: float) -> dict:
#     """Calculates eligible insurance payout after applying deductible and coverage cap."""
#     if claimed_amount_inr <= deductible_inr:
#         return {"eligible_payout_inr": 0, "reason": "Claimed amount is less than or equal to the deductible."}
#     net_claim = claimed_amount_inr - deductible_inr
#     approved_amount = min(net_claim, coverage_limit_inr)
#     return {
#         "claimed_amount_inr": claimed_amount_inr,
#         "deductible_applied_inr": deductible_inr,
#         "eligible_payout_inr": approved_amount,
#         "capped_by_limit": net_claim > coverage_limit_inr
#     }

# # --- 3. PDF Extraction Helper (PyMuPDF) ---
# def extract_pdf_text(uploaded_file) -> str:
#     doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#     pages = [page.get_text() for page in doc]
#     return "\n".join(pages).strip()

# # --- 4. Streamlit UI ---
# with st.sidebar:
#     st.subheader("Sample Policy Records (For Testing)")
#     st.dataframe(POLICIES_DB[["policy_id", "holder_name", "policy_type", "status", "coverage_limit_inr"]], hide_index=True)
#     st.info("Try referencing **POL-1001**, **POL-1002**, or **POL-1003** in your claim description or PDF.")

# col1, col2 = st.columns([1, 1])

# with col1:
#     st.subheader("1. Submit Claim Input")
#     uploaded_pdf = st.file_uploader(
#         "Upload Claim / Hospital Bill / Repair Estimate (PDF)",
#         type=["pdf"],
#         key="claim_pdf_uploader"
#     )
#     pdf_text = ""
#     if uploaded_pdf:
#         pdf_text = extract_pdf_text(uploaded_pdf)
#         with st.expander("Extracted PDF Text"):
#             st.text(pdf_text[:3000] if pdf_text else "No readable text found in PDF.")

#     claim_notes = st.text_area(
#         "Claim Description / Adjuster Notes",
#         height=160,
#         placeholder="Example: Policy ID POL-1001. Patient Rohan Deshmukh hospitalized for acute appendicitis surgery. Total hospital bill is ₹85,000. Discharge summary and itemized bills attached."
#     )

#     run_btn = st.button("🚀 Run Claim Triage Agent", type="primary", use_container_width=True)

# with col2:
#     st.subheader("2. Agent Decision & Audit Trail")
#     if run_btn:
#         if not pdf_text and not claim_notes.strip():
#             st.warning("Please upload a claim PDF or enter claim details first.")
#         else:
#             combined_input = f"Claim Notes:\n{claim_notes}\n\nExtracted Document Text:\n{pdf_text}"

#             system_instruction = """
#             You are ClaimPilot-AI, an expert insurance claims triage and adjudication agent.
#             Follow this workflow strictly:
# 1. Extract the Policy ID, claimant name, incident type, and claimed amount (in INR) from the input.
# 2. Call `lookup_policy` using the extracted Policy ID to verify policy status, coverage limits, deductible, and exclusions.
# 3. Check if the policy is Active and whether the incident violates any exclusions.
# 4. If eligible, call `calculate_payout` to compute the exact approved amount in INR.
# 5. Output a clean, structured Claim Adjudication Report with:
#   - **Decision**: APPROVED / PARTIALLY APPROVED / REJECTED / NEEDS MANUAL REVIEW
#   - **Policy Verification Summary**
#   - **Financial Breakdown (INR)**
#   - **Risk / Exclusion Check & Reasoning**
#             """

#             with st.spinner("Agent is inspecting documents and calling verification tools..."):
#                 models_to_try = ["gemini-3.1-flash-lite", "gemini-3.8-flash"]
#                 response = None
#                 last_error = None

#                 for model_name in models_to_try:
#                     for attempt in range(2):
#                         try:
#                             response = client.models.generate_content(
#                                 model=model_name,
#                                 contents=combined_input,
#                                 config=types.GenerateContentConfig(
#                                     system_instruction=system_instruction,
#                                     tools=[lookup_policy, calculate_payout],
#                                     temperature=0.1,
#                                 ),
#                             )
#                             breaksss
#                         except Exception as e:
#                             last_error = e
#                             if "503" in str(e) and attempt == 0:
#                                 time.sleep(1.5)
#                                 continue
#                             break
#                     if response is not None:
#                         break

#                 if response is not None:
#                     st.success(f"Triage Complete (Model: {model_name})")
#                     st.markdown(response.text)
#                 else:
#                     st.error(f"Agent Error: {last_error}")