import streamlit as st
import pandas as pd
import openai
import numpy as np

# استفاده از st.secrets برای دریافت کلید API OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# استفاده از st.session_state برای ناوبری بین صفحات
if "show_llm_page" not in st.session_state:
    st.session_state.show_llm_page = False
if "selected_transaction" not in st.session_state:
    st.session_state.selected_transaction = {}
if "llm_result" not in st.session_state:
    st.session_state.llm_result = ""

# (برای تست موقت، این خط را اضافه کنید تا محتویات st.secrets چاپ شود، سپس آن را کامنت کنید)
# st.write("Secrets:", st.secrets)

@st.cache_data
def load_normal_transactions():
    df = pd.read_csv("normal_transactions.csv")
    return df.to_dict(orient="records")

normal_transactions = load_normal_transactions()

dispute_transactions = [
    {"id": "D001", "description": "I never authorized this purchase of $300 at Amazon. This is fraudulent."},
    {"id": "D002", "description": "Charged twice for a purchase on eBay. Please refund one."},
    {"id": "D003", "description": "Unrecognized charge from ABC Electronics store, amount $250."},
    {"id": "D004", "description": "I was charged for an item I never received from AliExpress."},
    {"id": "D005", "description": "I canceled my subscription but still got charged again. Please fix it."},
]

def predict_fraud(transaction):
    amount = transaction["amount"]
    if amount > 300:
        prob = 0.85
        label = "Fraud"
    elif 150 <= amount <= 300:
        prob = 0.65
        label = "Dispute Transaction"
    else:
        prob = 0.20
        label = "Not Fraud"
    return label, prob

def analyze_dispute_with_llm(transaction_info):
    dispute_text = (
        f"Transaction ID: {transaction_info['id']}\n"
        f"Amount: ${transaction_info['amount']}\n"
        f"Merchant: {transaction_info['merchant']}\n"
        f"Location: {transaction_info['location']}\n"
        f"Time: {transaction_info['time']}\n"
        f"Device: {transaction_info['device']}\n"
        f"Past Spending: ${transaction_info['past_spending']}\n\n"
        "This transaction has been flagged as 'Dispute Transaction'. The customer reported an issue with it."
    )
    
    # اضافه کردن few-shot examples برای High, Medium, و Low risk
    prompt = f"""
You are a specialized Fraud Dispute Analysis assistant.

Here are some examples:

Example 1 (High Risk):
Original Dispute Text: I never authorized this purchase of $500 at StoreA.
Intent Category: Unauthorized/Fraud
Fraud Risk Score: High
AI-generated Summary: The transaction is highly suspicious due to the large amount and unfamiliar location.
Recommendation: Escalate the case to the internal fraud team.

Example 2 (Low Risk):
Original Dispute Text: I accidentally left my subscription active and got charged $20.
Intent Category: User Error
Fraud Risk Score: Low
AI-generated Summary: The user admits the charge, indicating it is not fraudulent.
Recommendation: Inform the user about cancellation procedures.

Example 3 (Medium Risk):
Original Dispute Text: I noticed a charge of $150 at an unfamiliar merchant.
Intent Category: Unclear
Fraud Risk Score: Medium
AI-generated Summary: The transaction has some risk due to the unknown merchant, but the amount is moderate.
Recommendation: Contact the customer for additional details.

Now, analyze the following disputed transaction details:

{dispute_text}

Tasks:
1. Display the Original Dispute Text as provided above.
2. Identify the Intent Category (e.g., 'Unauthorized/Fraud', 'Merchant Error', 'User Error', or 'Unclear').
3. Assess the Fraud Risk Score (High, Medium, or Low) with a short explanation.
4. Generate a concise, analyst-friendly summary.
5. Provide a recommended action for the fraud analyst (e.g., escalate, refund, contact customer, etc.).

IMPORTANT:
- You MUST strictly follow the output format.
- If any section is unknown, produce a placeholder.

Output format (strictly):
Original Dispute Text: <...>
Intent Category: <...>
Fraud Risk Score: <...>
AI-generated Summary: <...>
Recommendation: <...>
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

def show_llm_page():
    st.header("Dispute Transaction Analysis via AI Assistant")
    tx = st.session_state.selected_transaction
    llm_result = st.session_state.get("llm_result", "")
    
    # Debug: نمایش خروجی خام مدل (برای تست؛ سپس می‌توانید این خط را کامنت کنید)
    st.write("Debug LLM result (raw):")
    st.write(llm_result)
    
    lines = llm_result.splitlines()
    original_text = ""
    intent = ""
    risk = ""
    summary = ""
    recommendation = ""
    for line in lines:
        if line.startswith("Original Dispute Text:"):
            original_text = line.split(":", 1)[1].strip()
        elif line.startswith("Intent Category:"):
            intent = line.split(":", 1)[1].strip()
        elif line.startswith("Fraud Risk Score:"):
            risk = line.split(":", 1)[1].strip()
        elif line.startswith("AI-generated Summary:"):
            summary = line.split(":", 1)[1].strip()
        elif line.startswith("Recommendation:"):
            recommendation = line.split(":", 1)[1].strip()
    
    with st.expander("Original Dispute Text"):
        st.write(original_text if original_text else "No text provided.")
    
    st.markdown(f"**Intent Category:** {intent}")
    
    risk_color = "green"
    if "High" in risk:
        risk_color = "red"
    elif "Medium" in risk:
        risk_color = "yellow"
    st.markdown(
        f"**Fraud Risk Score:** <span style='color:white;background-color:{risk_color};padding:4px;border-radius:4px;'>{risk}</span>",
        unsafe_allow_html=True
    )
    st.markdown(f"**AI-generated Summary:** {summary}")
    st.markdown(f"**Recommendation:** {recommendation}")
    
    if st.button("Back to Main Page"):
        st.session_state.show_llm_page = False
        st.write("Please refresh the page manually to return to the main page.")

def show_main_page():
    st.title("Fraud Detection System - Hybrid Pipeline")
    st.sidebar.title("Settings")
    st.sidebar.write("Model: XGBoost Simulation")
    
    st.header("Normal Transaction Analysis (XGBoost Simulation)")
    for transaction in normal_transactions:
        st.write("---")
        cols = st.columns([3, 2])
        with cols[0]:
            st.markdown(f"**Transaction ID:** {transaction['id']}")
            st.write(f"**Merchant:** {transaction['merchant']}")
            st.write(f"**Amount:** ${transaction['amount']}")
            st.write(f"**Location:** {transaction['location']}")
            st.write(f"**Time:** {transaction['time']}")
            st.write(f"**Device:** {transaction['device']}")
            st.write(f"**Past Spending:** ${transaction['past_spending']}")
        with cols[1]:
            label, prob = predict_fraud(transaction)
            st.markdown(f"**XGBoost Label:** {label}")
            st.markdown(f"**Probability:** {prob:.2f}")
            if label == "Dispute Transaction":
                if st.button(f"Analyze with AI Assistant - {transaction['id']}", key=f"btn_{transaction['id']}"):
                    st.session_state.selected_transaction = transaction
                    llm_output = analyze_dispute_with_llm(transaction)
                    st.session_state.llm_result = llm_output
                    st.session_state.show_llm_page = True
    st.write("---")
    st.write("Please refresh the page manually to return to the main page.")

def main():
    if st.session_state.show_llm_page:
        show_llm_page()
    else:
        show_main_page()

if __name__ == "__main__":
    main()
