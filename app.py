import os
import re
import io
import time
import uuid
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NEW: Groq client
try:
    from groq import Groq
except ImportError:
    Groq = None

# ========== Helpers ==========

SPAMMY_WORDS = {
    "free", "winner", "guarantee", "risk-free", "act now", "trial", "click",
    "urgent", "exclusive", "limited", "amazing", "buy", "cheap", "lowest price"
}

DEFAULT_OPENERS = {
    "Concise": "I’m reaching out about the {role} opening—my background in {top_skills_str} lines up well with what you need.",
    "Warm": "Hope you’re doing well! I came across the {role} role at {company} and it immediately resonated with my recent work in {top_skills_str}.",
    "Direct": "I’m applying for {role} at {company}. I’ve delivered outcomes in {top_skills_str} and can add value quickly."
}

DEFAULT_CTAS = {
    "15-min chat": "Would you be open to a quick 15-minute chat this week?",
    "Referral ask": "If you’re not the right contact, could you point me to the hiring manager for {role}?",
    "Formal apply": "I’ve applied via the portal; happy to share more context or examples if helpful."
}

TONE_MAP = {
    "Professional": {"adjs": ["relevant", "measurable", "impactful"], "emoji": False},
    "Friendly": {"adjs": ["practical", "hands-on", "collaborative"], "emoji": True},
    "Crisp": {"adjs": ["lean", "focused", "results-driven"], "emoji": False},
    "Confident": {"adjs": ["high-impact", "owner-mindset", "fast-moving"], "emoji": False}
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
    jd_set = [kw for kw in jd_kw if kw in resume_kw]
    # keep variety: dedupe by stem-ish
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


def bulletize_achievements(achievements: str, max_bullets=3):
    lines = [normalize_text(x) for x in achievements.split("\n") if normalize_text(x)]
    # keep the strongest-looking lines first (has % or numbers)
    scored = []
    for ln in lines:
        score = 0
        if re.search(r"\b\d", ln):
            score += 1
        if "%" in ln:
            score += 1
        if re.search(r"\b(improv|increase|reduce|cut|boost|ship|deploy|launched)\b", ln, re.I):
            score += 1
        scored.append((score, ln))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:max_bullets]]


def spam_score(text: str) -> int:
    t = text.lower()
    return sum(1 for w in SPAMMY_WORDS if w in t)


def smart_wrap(s: str, max_len: int):
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def subject_variants(name, role, company, top_skill):
    base = [
        f"{role} @ {company}",
        f"{role} — {top_skill}",
        f"{role}: 60-sec intro ({name})",
        f"Quick note re: {role} at {company}",
        f"{company} • {role} • relevant experience",
    ]
    # uniqueness by ordering
    return list(dict.fromkeys(base))[:5]


def assemble_email(
    name,
    your_role,
    company,
    role,
    top_skills,
    achievements,
    opener_style,
    cta_style,
    tone,
    length,
):
    top_skills_str = ", ".join(top_skills[:4]) if top_skills else "the role’s core requirements"
    opener = DEFAULT_OPENERS[opener_style].format(
        role=role, company=company, top_skills_str=top_skills_str
    )
    cta = DEFAULT_CTAS[cta_style].format(role=role)
    adjs = TONE_MAP[tone]["adjs"]
    emoji = " 😊" if TONE_MAP[tone]["emoji"] else ""

    ach_lines = bulletize_achievements(achievements or "")
    if not ach_lines and top_skills:
        ach_lines = [
            f"Shipped projects using {top_skills[0]} and {top_skills[1] if len(top_skills) > 1 else ''}."
        ]
    bullets = "\n".join([f"- {ln}" for ln in ach_lines])

    value_line = (
        f"I focus on {adjs[0]} results and {adjs[1]} execution, and I’m comfortable owning ambiguous problems{emoji}"
    )

    body = f"""{opener}

Here’s why I think I’m a fit:
{bullets}

{value_line}
{cta}
"""
    if length == "Short (~90–120 words)":
        body = smart_wrap(body, 900)  # rough cap

    return body.strip()


# ========== Groq LLM integration ==========

def get_groq_client():
    # Prefer Streamlit secrets, fall back to env var
    api_key = None
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")

    if not api_key or Groq is None:
        return None, api_key

    return Groq(api_key=api_key), api_key


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
    """Call Groq to generate one high-quality cold email."""
    client, api_key = get_groq_client()
    if client is None or not api_key:
        raise RuntimeError(
            "Groq client not available or GROQ_API_KEY not set. "
            "Set GROQ_API_KEY in Streamlit secrets or as an environment variable."
        )

    tone_desc = {
        "Professional": "professional, concise, and recruiter-friendly",
        "Friendly": "friendly, warm, but still professional",
        "Crisp": "very concise, to the point, no fluff",
        "Confident": "confident, impact-focused, but not arrogant",
    }.get(tone, "professional and concise")

    overlap_str = ", ".join(top_overlap) if top_overlap else "key skills from the job description"

    prompt = f"""
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

Requirements for the email:
- 120–180 words.
- Start with a strong, specific SUBJECT line ("Subject: ...") on the first line.
- In the body, include 2–3 short bullet points that tie the candidate's experience to the job description.
- Make it easy for a recruiter to skim quickly.
- End with a simple call to action asking for a short call or next steps.
- Do NOT invent fake companies or fake numbers; only rephrase what is plausible from the text.
Return ONLY the email text, starting with 'Subject:' on the first line.
"""

    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You write concise, recruiter-friendly cold emails for job applications.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        max_completion_tokens=512,
    )

    return chat.choices[0].message.content.strip()


# ========== UI ==========

st.set_page_config(page_title="Cold Email Generator", page_icon="📬", layout="wide")
st.title("📬 Cold Email Generator — Resume + JD + Groq LLM")

with st.sidebar:
    st.markdown("### Your Info")
    name = st.text_input("Your Name", value="Aswin K M")
    your_role = st.text_input("Your Title (short)", value="Data Science Student")
    email_address = st.text_input("Your Email (optional)", value="")
    phone = st.text_input("Phone (optional)", value="")
    linkedin = st.text_input("LinkedIn URL (optional)", value="")

    st.markdown("---")
    st.markdown("### Target Role")
    company = st.text_input("Company", value="Acme AI")
    role = st.text_input("Role", value="Machine Learning Intern")
    opener_style = st.selectbox("Opening Style (Template mode)", list(DEFAULT_OPENERS.keys()), index=0)
    cta_style = st.selectbox("CTA Style (Template mode)", list(DEFAULT_CTAS.keys()), index=0)
    tone = st.selectbox("Tone", list(TONE_MAP.keys()), index=0)
    length = st.selectbox(
        "Template Length", ["Short (~90–120 words)", "Standard (~140–200 words)"], index=1
    )
    variants_n = st.slider("Template Email Variants", 1, 5, 3)

st.markdown("#### Paste your resume text")
resume_text = st.text_area(
    "Resume / portfolio text (plain text works great)",
    height=220,
    placeholder="Paste your resume summary plus highlights here...",
)

st.markdown("#### Paste the job description")
jd_text = st.text_area("Job description text", height=220, placeholder="Paste the JD here...")

st.markdown("#### Your strongest achievements (one per line)")
achievements = st.text_area(
    "Tip: include metrics (percent, dollars, time saved), shipped projects, leadership wins.",
    height=150,
    placeholder=(
        "Improved model F1 from 0.71 → 0.83 on imbalanced dataset (AUC +12%).\n"
        "Deployed Streamlit app used by 200+ students; cut planning time by 40%.\n"
        "Built ETL that reduced manual reporting by 6 hrs/week."
    ),
)

colA, colB, colC = st.columns([1, 1, 1])

# ---------- Template (no LLM) generation ----------
with colA:
    if st.button("Analyze Fit and Generate Templates"):
        with st.spinner("Scoring match and crafting template variants..."):
            r_kw = top_keywords(resume_text, n=30)
            j_kw = top_keywords(jd_text, n=30)
            top_overlap = overlap_keywords(r_kw, j_kw, k=8)
            sim = cosine_match(resume_text, jd_text)

            st.session_state["r_kw"] = r_kw
            st.session_state["j_kw"] = j_kw
            st.session_state["top_overlap"] = top_overlap
            st.session_state["sim"] = sim

            subjs = subject_variants(
                name, role, company, top_overlap[0] if top_overlap else "relevant skills"
            )

            emails = []
            for i in range(variants_n):
                body = assemble_email(
                    name=name,
                    your_role=your_role,
                    company=company,
                    role=role,
                    top_skills=top_overlap if top_overlap else j_kw[:5],
                    achievements=achievements,
                    opener_style=opener_style,
                    cta_style=cta_style,
                    tone=tone,
                    length=length,
                )
                signature_extra = []
                if email_address:
                    signature_extra.append(email_address)
                if phone:
                    signature_extra.append(phone)
                if linkedin:
                    signature_extra.append(linkedin)
                sig = ("\n" + " · ".join(signature_extra)) if signature_extra else ""
                body = body + sig
                emails.append(
                    {"subject": subjs[min(i, len(subjs) - 1)], "body": body}
                )

            st.session_state["emails"] = emails

with colB:
    if st.button("Export All Templates as CSV"):
        emails = st.session_state.get("emails", [])
        if not emails:
            st.warning("Generate template emails first.")
        else:
            df = pd.DataFrame(emails)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV", csv, file_name="cold_emails_templates.csv", mime="text/csv"
            )

with colC:
    if st.button("Export All Templates as .txt bundle"):
        emails = st.session_state.get("emails", [])
        if not emails:
            st.warning("Generate template emails first.")
        else:
            buffer = io.StringIO()
            for i, e in enumerate(emails, 1):
                buffer.write(
                    f"=== Email {i} (Template) ===\nSubject: {e['subject']}\n\n{e['body']}\n\n"
                )
            st.download_button(
                "Download TXT",
                data=buffer.getvalue().encode("utf-8"),
                file_name="cold_emails_templates.txt",
                mime="text/plain",
            )

st.markdown("---")

# ---------- Groq LLM generation ----------
if st.button("✨ Generate with Groq LLM"):
    if not resume_text or not jd_text:
        st.warning("Paste both your resume and the job description first.")
    else:
        r_kw = top_keywords(resume_text, n=30)
        j_kw = top_keywords(jd_text, n=30)
        top_overlap = overlap_keywords(r_kw, j_kw, k=8)

        try:
            with st.spinner("Calling Groq to draft your cold email..."):
                llm_email = generate_email_with_groq(
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
            st.session_state["groq_email"] = llm_email
        except Exception as e:
            st.error(f"Groq error: {e}")

# ========== Results Panels ==========

sim = st.session_state.get("sim")
if sim is not None:
    fit_pct = int(round(sim * 100))
    st.subheader(f"Match Score: {fit_pct}%")
    st.caption(
        "This is a simple TF-IDF cosine similarity between your resume and the JD. Use it as a rough guide."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top JD Keywords**")
        st.write(", ".join(st.session_state.get("j_kw", [])[:15]) or "—")
    with c2:
        st.markdown("**Resume ↔ JD Overlap**")
        overlap = st.session_state.get("top_overlap", [])
        st.write(", ".join(overlap) if overlap else "—")

emails = st.session_state.get("emails", [])
if emails:
    st.subheader("Template-based Generated Emails")
    for i, e in enumerate(emails, 1):
        sscore = spam_score(e["body"])
        with st.expander(
            f"Template Email {i}: {e['subject']} (spam-word hits: {sscore})",
            expanded=(i == 1),
        ):
            st.code(f"Subject: {e['subject']}\n\n{e['body']}")

groq_email = st.session_state.get("groq_email")
if groq_email:
    st.subheader("✨ Groq LLM Email")
    sscore = spam_score(groq_email)
    with st.expander(f"Groq Email (spam-word hits: {sscore})", expanded=True):
        st.code(groq_email)
