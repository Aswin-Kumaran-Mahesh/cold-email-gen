import os
import re
import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= Helpers =================

SPAMMY_WORDS = {
    "free", "winner", "guarantee", "risk-free", "act now", "trial", "click",
    "urgent", "exclusive", "limited", "amazing", "buy", "cheap", "lowest price"
}

TONE_DESC = {
    "Professional": "professional, concise, and recruiter-friendly",
    "Friendly": "friendly, warm, but still professional",
    "Crisp": "very concise, to-the-point, no fluff",
    "Confident": "confident, impact-focused, but not arrogant",
}


def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def top_keywords(text, n=20):
    text = normalize_text(text)
    if not text:
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform([text])
    feature_names = np.array(vec.get_feature_names_out())
    scores = X.toarray()[0]
    top_idx = np.argsort(scores)[::-1][:n]
    return [feature_names[i] for i in top_idx]


def overlap_keywords(resume_kw, jd_kw, k=8):
    jd_set = [k for k in jd_kw if k in resume_kw]
    seen = set()
    out = []
    for kw in jd_set:
        stem = kw.lower().split()[0]
        if stem not in seen:
            seen.add(stem)
            out.append(kw)
        if len(out) >= k:
            break
    return out


def cosine_match(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform([a, b])
    return float(cosine_similarity(X[0], X[1])[0, 0])


def spam_score(text: str) -> int:
    t = text.lower()
    return sum(1 for w in SPAMMY_WORDS if w in t)


# ================= Groq LLM via HTTP =================

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # adjust if you prefer another model


def generate_email_with_groq(
    resume_text,
    jd_text,
    achievements,
    name,
    your_role,
    company,
    role,
    tone,
    top_overlap,
):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Set it in your environment or Streamlit secrets."
        )

    tone_desc = TONE_DESC.get(tone, "professional and concise")
    overlap_str = ", ".join(top_overlap) if top_overlap else "key skills from the job description"

    system_msg = (
        "You write concise, recruiter-friendly cold emails for job applications."
    )

    user_prompt = f"""
You are an expert career coach helping candidates write cold emails to recruiters and hiring managers.

Write ONE cold email for this candidate:

Candidate name: {name}
Candidate title: {your_role}

Company: {company}
Target role: {role}

Tone style: {tone_desc}

Resume (raw text):
{resume_text}

Job description (raw text):
{jd_text}

Strongest achievements (raw, one per line):
{achievements}

Skills that overlap between resume and JD:
{overlap_str}

Requirements:
- 120–180 words.
- Start with a strong, specific SUBJECT line ("Subject: ...") on the first line.
- In the body, include 2–3 short bullet points that tie the candidate's experience to the job description.
- Make it easy for a recruiter to skim quickly.
- End with a simple call-to-action asking for a short call or next steps.
- Do NOT invent obviously fake companies or fake metrics; only rephrase plausible content.

Return ONLY the email text, starting with 'Subject:' on the first line.
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.6,
        "max_tokens": 512,
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected Groq response format: {e}, raw: {data}")


# ================= UI =================

st.set_page_config(page_title="Cold Email Generator", page_icon="📬", layout="wide")
st.title("📬 Groq-Powered Cold Email Generator")

with st.sidebar:
    st.markdown("### Your Info")
    name = st.text_input("Your Name", value="Aswin K M")
    your_role = st.text_input("Your Title (short)", value="Data Science Student")
    email_address = st.text_input("Your Email (optional)", value="")
    phone = st.text_input("Phone (optional)", value="")
    linkedin = st.text_input("LinkedIn URL (optional)", value="")

    st.markdown("---")
    st.markdown("### Target Role")
    company = st.text_input("Company", value="gagaAI")
    role = st.text_input("Role", value="Cloud Computing Intern")
    tone = st.selectbox("Tone", list(TONE_DESC.keys()), index=0)

st.markdown("#### Paste your resume text")
resume_text = st.text_area(
    "Resume / portfolio text",
    height=220,
    placeholder="Paste your resume summary + highlights here...",
)

st.markdown("#### Paste the job description")
jd_text = st.text_area(
    "Job description text",
    height=220,
    placeholder="Paste the JD here...",
)

st.markdown("#### Your strongest achievements (one per line)")
achievements = st.text_area(
    "Tip: include metrics (%, $, time saved), shipped projects, leadership wins.",
    height=150,
    placeholder=(
        "Improved model F1 from 0.71 → 0.83 on imbalanced dataset (AUC +12%).\n"
        "Deployed Streamlit app used by 200+ students; cut planning time by 40%.\n"
        "Built ETL that reduced manual reporting by 6 hrs/week."
    ),
)

if st.button("✨ Generate Email with Groq"):
    if not resume_text or not jd_text:
        st.warning("Paste both your resume and the job description first.")
    else:
        with st.spinner("Computing match score and asking Groq to draft your email..."):
            # compute similarity + overlap
            r_kw = top_keywords(resume_text, n=30)
            j_kw = top_keywords(jd_text, n=30)
            top_overlap = overlap_keywords(r_kw, j_kw, k=8)
            sim = cosine_match(resume_text, jd_text)

            st.session_state["sim"] = sim
            st.session_state["j_kw"] = j_kw
            st.session_state["top_overlap"] = top_overlap

            try:
                email_text = generate_email_with_groq(
                    resume_text=resume_text,
                    jd_text=jd_text,
                    achievements=achievements,
                    name=name,
                    your_role=your_role,
                    company=company,
                    role=role,
                    tone=tone,
                    top_overlap=top_overlap,
                )
                # add signature
                sig_parts = []
                if email_address:
                    sig_parts.append(email_address)
                if phone:
                    sig_parts.append(phone)
                if linkedin:
                    sig_parts.append(linkedin)
                signature = "\n" + " · ".join(sig_parts) if sig_parts else ""
                st.session_state["groq_email"] = email_text + signature
            except Exception as e:
                st.error(f"Groq error: {e}")

st.markdown("---")

# ======= Results: match score =======

sim = st.session_state.get("sim")
if sim is not None:
    fit_pct = int(round(sim * 100))
    st.subheader(f"Match Score: {fit_pct}%")
    st.caption("Simple TF-IDF cosine similarity between your resume and the JD.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top JD Keywords**")
        st.write(", ".join(st.session_state.get("j_kw", [])[:15]) or "—")
    with c2:
        st.markdown("**Resume ↔ JD Overlap**")
        overlap = st.session_state.get("top_overlap", [])
        st.write(", ".join(overlap) if overlap else "—")

# ======= Results: Groq email =======

groq_email = st.session_state.get("groq_email")
if groq_email:
    st.subheader("✨ Groq LLM Email")
    sscore = spam_score(groq_email)
    st.caption(f"Spam-word hits: {sscore}")
    st.code(groq_email)
