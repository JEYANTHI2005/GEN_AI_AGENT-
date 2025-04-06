

import streamlit as st
from transformers import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import re
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.badges import badge

tickets = [
    ("I want a refund for my broken TV", "Billing", 2),
    ("My internet is not working", "Tech Support", 5),
    ("Package not delivered yet", "Shipping", 3),
    ("Account login failed", "Tech Support", 4),
    ("Received wrong item", "Shipping", 3)
]

texts, labels, resolution_times = zip(*tickets)


try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
except:
    summarizer = lambda x, **kwargs: [{"summary_text": "(Summary model not available)"}]

try:
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")
except:
    sentiment_analyzer = lambda x: [{"label": "Neutral", "score": 0.5}]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
classifier = LogisticRegression().fit(X, labels)
regressor = RandomForestRegressor().fit(X.toarray(), resolution_times)

# Knowledge Base for Resolution Recommendation
knowledge_base = {
    "refund": "Please process a refund via the billing system.",
    "internet": "Restart the router and run a diagnostic scan.",
    "package": "Check courier tracking and escalate if delayed.",
    "login": "Reset password or check account lock.",
    "wrong item": "Initiate a return and send replacement."
}

st.set_page_config(page_title="NovaServe AI Support", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
        .main {background-color: #f5f7fa; padding: 30px; border-radius: 10px;}
        .stTextArea textarea {font-size: 18px; height: 150px;}
        .stButton>button {font-size: 18px; border-radius: 10px; padding: 10px 20px;}
    </style>
""", unsafe_allow_html=True)

st.title("💬 NovaServe - Smart AI Customer Support")
badge(type="github", name="NovaServe")
add_vertical_space(1)

st.markdown("###  AI-Enhanced Customer Query Analysis")
user_query = st.text_area("🔍 Enter customer query below:")

if st.button(" Analyze Query") and user_query:
    with st.spinner(" Running AI agents to analyze the query..."):
        summary = summarizer(user_query, max_length=30, min_length=10, do_sample=False)[0]['summary_text']
        sentiment = sentiment_analyzer(user_query)[0]
        query_vec = vectorizer.transform([user_query])
        department = classifier.predict(query_vec)[0]
        time_estimate = regressor.predict(query_vec.toarray())[0]

        recommendation = "Escalate to human agent for review."
        for key in knowledge_base:
            if re.search(rf"\\b{re.escape(key)}\\b", user_query.lower()):
                recommendation = knowledge_base[key]
                break

    with st.container():
        st.subheader("📡 AI Agent Outputs")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**📝 Summary:** {summary}")
            st.markdown(f"**📊 Sentiment:** {sentiment['label']} (Score: {sentiment['score']:.2f})")

        with col2:
            st.markdown(f"**📌 Routed to:** `{department} Department`")
            st.markdown(f"**⏱ Estimated Resolution Time:** `{round(time_estimate, 1)} hours`")

        add_vertical_space(1)
        st.success(f"💡 **Recommended Resolution:** {recommendation}")

    st.markdown("### 🤔 Was this helpful?")
    feedback = st.radio("", ["👍 Yes", "👎 No"], horizontal=True)
    if feedback:
        st.toast("Thanks for your feedback!", icon="👏")

    st.balloons()
