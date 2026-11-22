import html
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import fitz  # pymupdf
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DB_PATH = Path("data/claims.db")

DEFAULT_POLICIES = [
    (
        "POL-1001",
        "Rohan Deshmukh",
        "Comprehensive Health",
        "Active",
        500000,
        5000,
        1,
        "Cosmetic surgery, experimental treatments, pre-existing conditions within 24 months",
    ),
    (
        "POL-1002",
        "Sneha Kulkarni",
        "Zero-Dep Motor Comprehensive",
        "Active",
        800000,
        2000,
        1,
        "Drunk driving, driving without valid license, consequential engine water damage",
    ),
    (
        "POL-1003",
        "Amit Patil",
        "Comprehensive Health",
        "Lapsed",
        300000,
        5000,
        0,
        "All claims invalid while policy status is Lapsed",
    ),
]


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS policies (
                policy_id TEXT PRIMARY KEY,
                holder_name TEXT NOT NULL,
                policy_type TEXT NOT NULL,
                status TEXT NOT NULL,
                coverage_limit_inr REAL NOT NULL,
                deductible_inr REAL NOT NULL,
                waiting_period_completed INTEGER NOT NULL,
                exclusions TEXT NOT NULL
            )
            """
        )
        count = conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
        if count == 0:
            conn.executemany(
                """
                INSERT INTO policies (
                    policy_id, holder_name, policy_type, status,
                    coverage_limit_inr, deductible_inr, waiting_period_completed, exclusions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                DEFAULT_POLICIES,
            )
        conn.commit()


def load_policies_df() -> pd.DataFrame:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM policies ORDER BY policy_id", conn)


st.set_page_config(
    page_title="ClaimPilot-AI | Insurance Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling (works in both light and dark themes)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px;}
    #MainMenu, footer {visibility: hidden;}

    .cp-hero {
        padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1.4rem;
        background: linear-gradient(120deg, #0f2a43 0%, #14506b 100%);
        color: #f2f7fa;
    }
    .cp-hero h1 {margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.01em; color: #ffffff;}
    .cp-hero p {margin: .45rem 0 0 0; font-size: .95rem; line-height: 1.5; color: #cfe0ea; max-width: 760px;}

    .cp-section {font-size: 1.15rem; font-weight: 650; margin: 0 0 .6rem 0;}

    /* Sidebar policy table */
    .cp-table {width: 100%; border-collapse: collapse; font-size: .82rem;}
    .cp-table th {
        text-align: left; padding: .45rem .5rem; font-weight: 600;
        border-bottom: 1px solid rgba(128,128,128,.35); white-space: nowrap;
    }
    .cp-table td {padding: .5rem .5rem; border-bottom: 1px solid rgba(128,128,128,.18); vertical-align: top;}
    .cp-table td.id {font-weight: 600; white-space: nowrap;}
    .cp-table td.num {text-align: right; white-space: nowrap;}
    .cp-chip {
        display: inline-block; padding: .08rem .55rem; border-radius: 999px;
        font-size: .74rem; font-weight: 600;
    }
    .chip-active {background: rgba(22,163,74,.16); color: #16a34a;}
    .chip-lapsed {background: rgba(220,38,38,.16); color: #dc2626;}

    /* Decision banner */
    .cp-banner {
        display: flex; align-items: center; gap: .9rem; padding: 1rem 1.2rem;
        border-radius: 12px; border: 1px solid; margin-bottom: 1rem;
    }
    .cp-banner .icon {font-size: 1.8rem; line-height: 1;}
    .cp-banner .title {font-size: 1.25rem; font-weight: 700; line-height: 1.2;}
    .cp-banner .sub {font-size: .85rem; opacity: .85; margin-top: .15rem;}
    .banner-approved {background: rgba(22,163,74,.12); border-color: rgba(22,163,74,.55); color: #15803d;}
    .banner-flagged  {background: rgba(217,119,6,.12); border-color: rgba(217,119,6,.55); color: #b45309;}
    .banner-rejected {background: rgba(220,38,38,.12); border-color: rgba(220,38,38,.55); color: #b91c1c;}

    /* Step breakdown */
    .cp-step {
        display: flex; gap: .8rem; padding: .7rem .9rem; margin-bottom: .5rem;
        border-radius: 10px; border: 1px solid rgba(128,128,128,.25);
        border-left-width: 5px;
    }
    .cp-step .ico {font-size: 1.15rem; line-height: 1.4;}
    .cp-step .name {font-weight: 650; font-size: .93rem;}
    .cp-step .detail {font-size: .85rem; opacity: .85; margin-top: .1rem; line-height: 1.45;}
    .step-pass {border-left-color: #16a34a;}
    .step-warn {border-left-color: #d97706;}
    .step-fail {border-left-color: #dc2626;}
    .step-info {border-left-color: #64748b;}

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: .6rem .9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="cp-hero">
        <h1>🛡️ ClaimPilot-AI</h1>
        <p><b>Autonomous Insurance Claim Triage Agent.</b> Upload a claim bill, medical record or repair
        estimate (PDF or image), or type the claim details. The agent verifies the policy, checks exclusions
        and recommends an adjudication.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "your_actual_api_key_here":
    st.error("Please add your valid GEMINI_API_KEY inside the .env file and refresh.")
    st.stop()

client = genai.Client(api_key=api_key)

MODELS_TO_TRY = ["gemini-3.1-flash-lite", "gemini-3.8-flash"]

# --- 1. Agent Tools (Python functions Gemini can call autonomously) ---
def lookup_policy(policy_id: str) -> dict:
    """Fetches policy details, coverage limit, deductible, status, and exclusions by Policy ID."""
    init_db()
    clean_id = policy_id.strip().upper()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM policies WHERE UPPER(policy_id) = ?",
            (clean_id,),
        ).fetchone()
    if row is None:
        return {"found": False, "message": f"No policy found for ID {policy_id}."}
    policy = dict(row)
    policy["waiting_period_completed"] = bool(policy["waiting_period_completed"])
    return {"found": True, "policy": policy}


def calculate_payout(claimed_amount_inr: float, coverage_limit_inr: float, deductible_inr: float) -> dict:
    """Calculates eligible insurance payout after applying deductible and coverage cap."""
    if claimed_amount_inr <= deductible_inr:
        return {"eligible_payout_inr": 0, "reason": "Claimed amount is less than or equal to the deductible."}
    net_claim = claimed_amount_inr - deductible_inr
    approved_amount = min(net_claim, coverage_limit_inr)
    return {
        "claimed_amount_inr": claimed_amount_inr,
        "deductible_applied_inr": deductible_inr,
        "eligible_payout_inr": approved_amount,
        "capped_by_limit": net_claim > coverage_limit_inr,
    }


# --- 2. Document Extraction Helpers (PyMuPDF first, Gemini vision as fallback) ---
MIN_TEXT_CHARS = 50          # below this a PDF is treated as scanned
MAX_SCAN_PAGES = 5           # pages sent to vision for scanned PDFs

VISION_PROMPT = (
    "You are reading an insurance claim document (hospital bill, discharge summary, repair estimate, "
    "invoice or claim form). Extract every claim-relevant detail exactly as written and return clean "
    "plain text using these labelled fields where present: Policy ID, Claimant / Patient name, "
    "Provider / Hospital / Garage, Date(s), Incident or treatment description, Itemized charges, "
    "Total claimed amount (INR), Other notes. Do not invent or guess values; write 'Not found' for "
    "anything that is missing."
)


def extract_pdf_text(uploaded_file) -> str:
    doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
    pages = [page.get_text() for page in doc]
    return "\n".join(pages).strip()


def vision_extract(parts: list) -> str:
    """Sends image parts to Gemini (with the same model fallback) and returns extracted text."""
    last_error = None
    for model_name in MODELS_TO_TRY:
        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=parts + [VISION_PROMPT],
                    config=types.GenerateContentConfig(temperature=0.0),
                )
                return (resp.text or "").strip()
            except Exception as e:
                last_error = e
                if "503" in str(e) and attempt == 0:
                    time.sleep(1.5)
                    continue
                break
    raise RuntimeError(f"Vision extraction failed: {last_error}")


def process_upload(uploaded_file):
    """Returns (extracted_text, method_label)."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        text = extract_pdf_text(uploaded_file)
        if len(text) >= MIN_TEXT_CHARS:
            return text, "PDF text layer (PyMuPDF)"
        # Scanned PDF: render pages to images and read them with Gemini vision
        doc = fitz.open(stream=data, filetype="pdf")
        parts = []
        for i, page in enumerate(doc):
            if i >= MAX_SCAN_PAGES:
                break
            png = page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")
            parts.append(types.Part.from_bytes(data=png, mime_type="image/png"))
        return vision_extract(parts), "Scanned PDF read with Gemini vision"

    mime = "image/png" if name.endswith(".png") else "image/jpeg"
    part = types.Part.from_bytes(data=data, mime_type=mime)
    return vision_extract([part]), "Image read with Gemini vision"


# --- 3. Sample claims for quick testing ---
SAMPLE_CLAIMS = {
    "Active Health (POL-1001)": (
        "Policy ID POL-1001. Patient Rohan Deshmukh was hospitalized for acute appendicitis surgery. "
        "Total hospital bill is ₹85,000. Discharge summary and itemized bills attached."
    ),
    "Motor Exclusion (POL-1002)": (
        "Policy ID POL-1002. Sneha Kulkarni reports vehicle damage after a collision. The police report "
        "states the driver was under the influence of alcohol at the time of the accident. "
        "Repair estimate is ₹1,20,000."
    ),
    "Lapsed Policy (POL-1003)": (
        "Policy ID POL-1003. Amit Patil was admitted for treatment of a viral infection. "
        "Total hospital bill is ₹60,000. Please check eligibility."
    ),
}


def load_sample(text: str):
    st.session_state["claim_notes"] = text


# --- 4. Result parsing helpers (read-only; they do not change the agent) ---
def parse_tool_calls(response):
    """Collects tool results and arguments from the automatic function-calling history."""
    results, args = {}, {}
    history = getattr(response, "automatic_function_calling_history", None) or []
    for content in history:
        for part in (content.parts or []):
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                args[fc.name] = dict(fc.args or {})
            fr = getattr(part, "function_response", None)
            if fr and fr.name:
                resp = fr.response or {}
                results[fr.name] = resp.get("result", resp) if isinstance(resp, dict) else resp
    return results, args


def parse_decision(text: str, lookup):
    """Maps the agent's decision to one of: approved, flagged, rejected."""
    m = re.search(
        r"decision\W*(NEEDS MANUAL REVIEW|PARTIALLY APPROVED|APPROVED|REJECTED)",
        text or "", re.IGNORECASE,
    )
    label = m.group(1).upper() if m else None
    if label == "REJECTED":
        return "rejected", "Rejected"
    if label == "NEEDS MANUAL REVIEW":
        return "flagged", "Flagged for manual review"
    if label == "PARTIALLY APPROVED":
        return "approved", "Approved (partial)"
    if label == "APPROVED":
        return "approved", "Approved"
    # Fallback when the report has no parsable decision line
    policy = (lookup or {}).get("policy") if isinstance(lookup, dict) else None
    if policy and policy.get("status") != "Active":
        return "rejected", "Rejected"
    return "flagged", "Flagged for manual review"


def inr(value):
    try:
        return f"₹{float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"


def build_steps(tools, tool_args, decision_key):
    lookup = tools.get("lookup_policy")
    payout = tools.get("calculate_payout")
    pargs = tool_args.get("calculate_payout", {})
    policy = lookup.get("policy") if isinstance(lookup, dict) and lookup.get("found") else None
    steps = []

    # Step: policy status
    if lookup is None:
        steps.append(("warn", "Policy status", "Policy lookup was not performed by the agent."))
    elif not policy:
        steps.append(("fail", "Policy status", lookup.get("message", "No matching policy found.")))
    elif policy["status"] == "Active":
        steps.append(("pass", "Policy status",
                      f"{policy['policy_id']} ({policy['holder_name']}, {policy['policy_type']}) is Active."))
    else:
        steps.append(("fail", "Policy status",
                      f"{policy['policy_id']} ({policy['holder_name']}) is {policy['status']}, so claims are not payable."))

    # Step: exclusions
    if policy:
        excl = policy["exclusions"]
        if policy["status"] != "Active":
            steps.append(("fail", "Exclusions", f"Listed exclusions: {excl}"))
        elif decision_key == "rejected":
            steps.append(("fail", "Exclusions", f"Claim appears to fall under a listed exclusion. Listed exclusions: {excl}"))
        elif decision_key == "flagged":
            steps.append(("warn", "Exclusions", f"Needs a closer look. Listed exclusions: {excl}"))
        else:
            steps.append(("pass", "Exclusions", f"No listed exclusion triggered. Listed exclusions: {excl}"))
    else:
        steps.append(("info", "Exclusions", "Not evaluated because no policy was verified."))

    # Step: deductible
    if policy:
        ded = (payout or {}).get("deductible_applied_inr", policy["deductible_inr"])
        note = "applied to the claim" if payout and "deductible_applied_inr" in payout else "on this policy"
        steps.append(("info" if not payout else "pass", "Deductible", f"{inr(ded)} {note}."))
    else:
        steps.append(("info", "Deductible", "Not available."))

    # Step: coverage cap
    if policy:
        if payout and payout.get("capped_by_limit"):
            steps.append(("warn", "Coverage cap",
                          f"Coverage limit of {inr(policy['coverage_limit_inr'])} reached; payout was capped."))
        else:
            steps.append(("pass" if payout else "info", "Coverage cap",
                          f"Coverage limit is {inr(policy['coverage_limit_inr'])}. Claim is within the limit."
                          if payout else f"Coverage limit is {inr(policy['coverage_limit_inr'])}."))
    else:
        steps.append(("info", "Coverage cap", "Not available."))

    # Step: payout
    if payout:
        amount = payout.get("eligible_payout_inr", 0)
        extra = f" {payout['reason']}" if payout.get("reason") else ""
        steps.append(("pass" if amount else "fail", "INR payout", f"Eligible payout: {inr(amount)}.{extra}"))
    elif decision_key == "rejected":
        steps.append(("fail", "INR payout", "₹0. The claim is not payable."))
    else:
        steps.append(("warn", "INR payout", "Payout was not calculated."))

    claimed = pargs.get("claimed_amount_inr", (payout or {}).get("claimed_amount_inr"))
    return steps, claimed, (payout or {}).get("eligible_payout_inr")


def build_report(result) -> str:
    icons = {"pass": "PASS", "warn": "REVIEW", "fail": "FAIL", "info": "INFO"}
    lines = [
        "# ClaimPilot-AI Triage Report",
        "",
        f"- **Generated:** {result['timestamp']}",
        f"- **Model:** {result['model']}",
        f"- **Input source:** {result['source']}",
        f"- **Outcome:** {result['label']}",
        f"- **Claimed amount:** {inr(result['claimed'])}",
        f"- **Eligible payout:** {inr(result['payout'])}",
        "",
        "## Step-by-step breakdown",
        "",
    ]
    for state, name, detail in result["steps"]:
        lines.append(f"- **{name}** [{icons[state]}]: {detail}")
    lines += ["", "## Agent report", "", result["text"], ""]
    return "\n".join(lines)


# --- 5. Streamlit UI ---
STATUS_META = {
    "approved": ("✅", "banner-approved", "All checks passed for this claim."),
    "flagged": ("⚠️", "banner-flagged", "A human adjuster should review this claim before any payout."),
    "rejected": ("❌", "banner-rejected", "This claim is not payable under the policy terms."),
}
STEP_ICONS = {"pass": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}

with st.sidebar:
    st.subheader("Sample Policy Records")
    policies_df = load_policies_df()
    rows = ""
    for _, p in policies_df.iterrows():
        chip = "chip-active" if p["status"] == "Active" else "chip-lapsed"
        rows += (
            f"<tr><td class='id'>{html.escape(p['policy_id'])}</td>"
            f"<td><span class='cp-chip {chip}'>{html.escape(p['status'])}</span></td>"
            f"<td class='num'>₹{int(p['coverage_limit_inr']):,}</td>"
            f"<td>{html.escape(p['policy_type'])}</td></tr>"
        )
    st.markdown(
        "<table class='cp-table'><thead><tr><th>Policy ID</th><th>Status</th>"
        "<th style='text-align:right'>Limit (INR)</th><th>Type</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )
    st.info("Reference **POL-1001**, **POL-1002**, or **POL-1003** in your claim description or document.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='cp-section'>1. Submit Claim Input</div>", unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader(
        "Upload claim, hospital bill or repair estimate (PDF, PNG, JPG, JPEG)",
        type=["pdf", "png", "jpg", "jpeg"],
        key="claim_pdf_uploader",
    )

    pdf_text = ""
    doc_method = None
    if uploaded_pdf:
        cache_key = f"{uploaded_pdf.name}-{uploaded_pdf.size}"
        if st.session_state.get("doc_cache_key") != cache_key:
            with st.spinner("Reading document..."):
                try:
                    text, method = process_upload(uploaded_pdf)
                    st.session_state["doc_cache_key"] = cache_key
                    st.session_state["doc_text"] = text
                    st.session_state["doc_method"] = method
                    st.session_state["doc_error"] = None
                except Exception as e:
                    st.session_state["doc_cache_key"] = cache_key
                    st.session_state["doc_text"] = ""
                    st.session_state["doc_method"] = None
                    st.session_state["doc_error"] = str(e)
        pdf_text = st.session_state.get("doc_text", "")
        doc_method = st.session_state.get("doc_method")

        if st.session_state.get("doc_error"):
            st.error(st.session_state["doc_error"])
        else:
            if doc_method:
                st.caption(f"Extraction method: {doc_method}")
            with st.expander("Extracted document text"):
                st.text(pdf_text[:3000] if pdf_text else "No readable text found in the document.")
            if uploaded_pdf.type and uploaded_pdf.type.startswith("image/"):
                st.image(uploaded_pdf, caption=uploaded_pdf.name, use_container_width=True)
    else:
        st.session_state.pop("doc_cache_key", None)

    st.markdown("**Quick samples**")
    sample_cols = st.columns(len(SAMPLE_CLAIMS))
    for i, (label, text) in enumerate(SAMPLE_CLAIMS.items()):
        sample_cols[i].button(
            label, key=f"sample_btn_{i}", on_click=load_sample, args=(text,), use_container_width=True
        )

    claim_notes = st.text_area(
        "Claim Description / Adjuster Notes",
        height=170,
        key="claim_notes",
        placeholder="Example: Policy ID POL-1001. Patient Rohan Deshmukh hospitalized for acute appendicitis surgery. Total hospital bill is ₹85,000. Discharge summary and itemized bills attached."
    )

    run_btn = st.button("🚀 Run Claim Triage Agent", type="primary", use_container_width=True, key="run_triage_btn")

if run_btn:
    if not pdf_text and not claim_notes.strip():
        st.session_state["triage_warning"] = "Please upload a claim document or enter claim details first."
        st.session_state.pop("triage_result", None)
        st.session_state.pop("triage_error", None)
    else:
        st.session_state.pop("triage_warning", None)
        st.session_state.pop("triage_error", None)

        combined_input = f"Claim Notes:\n{claim_notes}\n\nExtracted Document Text:\n{pdf_text}"

        system_instruction = """
        You are ClaimPilot-AI, an expert insurance claims triage and adjudication agent.
        Follow this workflow strictly:
1. Extract the Policy ID, claimant name, incident type, and claimed amount (in INR) from the input.
2. Call `lookup_policy` using the extracted Policy ID to verify policy status, coverage limits, deductible, and exclusions.
3. Check if the policy is Active and whether the incident violates any exclusions.
4. If eligible, call `calculate_payout` to compute the exact approved amount in INR.
5. Output a clean, structured Claim Adjudication Report with:
  - **Decision**: APPROVED / PARTIALLY APPROVED / REJECTED / NEEDS MANUAL REVIEW
  - **Policy Verification Summary**
  - **Financial Breakdown (INR)**
  - **Risk / Exclusion Check & Reasoning**
        """

        with col2:
            with st.spinner("Agent is inspecting documents and calling verification tools..."):
                response = None
                last_error = None
                used_model = None

                for model_name in MODELS_TO_TRY:
                    for attempt in range(2):
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=combined_input,
                                config=types.GenerateContentConfig(
                                    system_instruction=system_instruction,
                                    tools=[lookup_policy, calculate_payout],
                                    temperature=0.1,
                                ),
                            )
                            break
                        except Exception as e:
                            last_error = e
                            if "503" in str(e) and attempt == 0:
                                time.sleep(1.5)
                                continue
                            break
                    if response is not None:
                        used_model = model_name
                        break

        if response is not None:
            text = response.text or ""
            tools, tool_args = parse_tool_calls(response)
            decision_key, label = parse_decision(text, tools.get("lookup_policy"))
            steps, claimed, payout_amt = build_steps(tools, tool_args, decision_key)
            source_bits = []
            if pdf_text and doc_method:
                source_bits.append(doc_method)
            if claim_notes.strip():
                source_bits.append("typed claim notes")
            result = {
                "text": text,
                "model": used_model,
                "decision_key": decision_key,
                "label": label,
                "steps": steps,
                "claimed": claimed,
                "payout": payout_amt,
                "source": " + ".join(source_bits) or "n/a",
                "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
            }
            st.session_state["triage_result"] = result
        else:
            st.session_state["triage_error"] = str(last_error)
            st.session_state.pop("triage_result", None)

with col2:
    st.markdown("<div class='cp-section'>2. Agent Decision & Audit Trail</div>", unsafe_allow_html=True)

    result = st.session_state.get("triage_result")

    if st.session_state.get("triage_warning"):
        st.warning(st.session_state["triage_warning"])
    elif st.session_state.get("triage_error"):
        st.error(f"Agent Error: {st.session_state['triage_error']}")
    elif not result:
        st.info("Add a claim on the left or pick a quick sample, then run the agent. The decision and audit trail appear here.")
    else:
        icon, banner_cls, sub = STATUS_META[result["decision_key"]]
        st.markdown(
            f"<div class='cp-banner {banner_cls}'><div class='icon'>{icon}</div>"
            f"<div><div class='title'>{html.escape(result['label'])}</div>"
            f"<div class='sub'>{sub} Model: {html.escape(result['model'])}</div></div></div>",
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        m1.metric("Claimed amount", inr(result["claimed"]))
        m2.metric("Eligible payout", inr(result["payout"]) if result["payout"] is not None else "—")

        st.markdown("**Step-by-step breakdown**")
        for state, name, detail in result["steps"]:
            st.markdown(
                f"<div class='cp-step step-{state}'><div class='ico'>{STEP_ICONS[state]}</div>"
                f"<div><div class='name'>{html.escape(name)}</div>"
                f"<div class='detail'>{html.escape(detail)}</div></div></div>",
                unsafe_allow_html=True,
            )

        with st.expander("Full agent report", expanded=False):
            st.markdown(result["text"])

        st.download_button(
            "⬇️ Download triage report (.md)",
            data=build_report(result),
            file_name=f"claimpilot_triage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
            key="download_report_btn",
        )