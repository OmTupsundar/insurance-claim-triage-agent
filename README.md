# ClaimPilot-AI: Autonomous Insurance Claim Triage Agent

An AI agent built with Python, Streamlit, PyMuPDF, and Google Gemini to automate insurance claims triage, policy verification, exclusion checking, and payout calculation.

## Insurance Use Case
Insurance operations handle high volumes of claim descriptions, repair estimates, and hospital bills. **ClaimPilot-AI** extracts key claim details from adjuster notes and uploaded PDFs, verifies policy status and exclusions against policy records, computes eligible payouts after deductibles and coverage limits, and generates a structured adjudication report.

## Architecture & Workflow
1. **Document & Input Ingestion (`app.py`):** Accepts claim descriptions and extracts text from uploaded PDFs using PyMuPDF (`fitz`).
2. **Policy Lookup Tool (`lookup_policy`):** Queries policy records (status, coverage limit in INR, deductible in INR, and exclusions) by Policy ID.
3. **Payout Calculation Tool (`calculate_payout`):** Computes eligible payout in INR after applying deductibles and coverage caps.
4. **Agent Reasoning & Triage Report:** Uses Gemini function calling to inspect policy eligibility, flag exclusions or lapsed policies, and generate a structured adjudication summary.

## Setup & Run Locally

1. **Clone the repository and create a virtual environment:**
   ```bash
   git clone <your-public-github-repo-url>
   cd ClaimPilot-AI
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   * **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
   * **Windows (CMD):** `venv\Scripts\activate.bat`
   * **Linux / macOS:** `source venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Sample Test Cases
* **Active Health Claim (`POL-1001`):** Hospitalization and surgery claim within coverage limits (`APPROVED`).
* **Motor Exclusion Check (`POL-1002`):** Claim involving waterlogging / consequential engine damage exclusion (`FLAGGED / REJECTED`).
* **Lapsed Policy Check (`POL-1003`):** Claim submitted against a lapsed policy (`REJECTED`).