# ANON: Tata AutoComp Innovation Feedback Web App (Streamlit Version)

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bertopic import BERTopic
import random

# --- PAGE CONFIG ---
st.set_page_config(page_title="ANON Dashboard | Tata AutoComp", layout="wide")

# --- HEADER / BRANDING ---
st.markdown("""
<div style="background-color:#0066b3;padding:10px 20px;display:flex;align-items:center;">
  <img src="https://www.tataautocomp.com/wp-content/themes/tataautocomp/images/logo.svg" style="height:40px;margin-right:20px;">
  <h1 style="color:white;font-family:Arial, sans-serif;font-size:24px;">ANON: Tata AutoComp Innovation Feedback</h1>
</div>
""", unsafe_allow_html=True)

# --- SAMPLE DATA ---
if 'ideas_df' not in st.session_state:
    sample_ideas = [
        "Cross-department collaboration is too slow and manual.",
        "Lack of clear promotion pathways affects motivation.",
        "Need better documentation for EV battery assembly process.",
        "Lunch break timing overlaps reduce assembly line efficiency.",
        "Inconsistent email updates cause confusion between plants.",
        "Too many unnecessary meetings slow down project sprints.",
        "Pressure during year-end affects mental health.",
        "No clarity on skills needed for internal role switches."
    ]
    sample_moods = ["😠", "🙂", "🤔", "😐", "😊"]
    st.session_state.ideas_df = pd.DataFrame([{
        'text': text,
        'mood': random.choice(sample_moods),
        'timestamp': datetime.now(),
        'status': random.choice(["🟡 New", "🟢 Reviewed", "🔴 Addressed"]),
    } for text in sample_ideas])

ideas_df = st.session_state.ideas_df

# --- SIDEBAR NAV ---
page = st.sidebar.radio("📂 Navigation", ["📊 Dashboard", "🧩 Submit Idea", "🧠 AI Clusters", "🔄 Tone Translator", "📥 Export Data"])

# --- DASHBOARD ---
if page == "📊 Dashboard":
    st.title("📊 Mood Dashboard")

    st.subheader("Mood Distribution")
    mood_counts = ideas_df['mood'].value_counts()
    fig = px.bar(x=mood_counts.index, y=mood_counts.values, labels={'x': 'Mood', 'y': 'Count'}, color_discrete_sequence=['#0066b3'])
    st.plotly_chart(fig)

    st.subheader("WordCloud")
    text = " ".join(ideas_df["text"])
    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("No text to show.")

# --- SUBMIT IDEA ---
elif page == "🧩 Submit Idea":
    st.title("🧩 Submit an Idea")
    with st.form("submit_form"):
        text_input = st.text_area("Enter your idea anonymously:")
        mood = st.selectbox("Mood", ["😠 Frustrated", "🙂 Hopeful", "🤔 Confused", "😐 Neutral", "😊 Excited"])
        submitted = st.form_submit_button("Submit")
        if submitted:
            if text_input.strip():
                new_row = {
                    "text": text_input.strip(),
                    "mood": mood,
                    "timestamp": datetime.now(),
                    "status": "🟡 New"
                }
                st.session_state.ideas_df = pd.concat([st.session_state.ideas_df, pd.DataFrame([new_row])], ignore_index=True)
                st.success("✅ Idea submitted successfully.")
                st.experimental_rerun()
            else:
                st.warning("Please enter a valid idea.")

# --- AI CLUSTERS ---
elif page == "🧠 AI Clusters":
    st.title("🧠 AI-Detected Topic Clusters")
    if not ideas_df.empty:
        topic_model = BERTopic(verbose=False)
        topics, _ = topic_model.fit_transform(ideas_df["text"])
        st.session_state.ideas_df['topic'] = topics

        topic_info = topic_model.get_topic_info()
        st.dataframe(topic_info.head())

        st.plotly_chart(topic_model.visualize_topics())

# --- TONE TRANSLATOR ---
elif page == "🔄 Tone Translator":
    st.title("🔄 Tone Translator")
    input_text = st.text_area("Engineering phrasing:")
    if st.button("Translate"):
        if input_text.strip():
            mgmt_tone = f"We acknowledge the concern: '{input_text}'. This will be reviewed for continuous improvement."
            eng_tone = f"'{input_text}' — this needs to be fixed now."
            st.markdown(f"**📢 Management Style:** {mgmt_tone}")
            st.markdown(f"**🔧 Engineering Style:** {eng_tone}")
        else:
            st.warning("Please enter a phrase to translate.")

# --- EXPORT DATA ---
elif page == "📥 Export Data":
    st.title("📥 Export All Data")
    st.dataframe(ideas_df)
    csv = ideas_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download as CSV", data=csv, file_name='anon_ideas.csv', mime='text/csv')
