# ❄️ Cold Email Generator

AI powered Streamlit app that generates personalized cold emails for recruiters, hiring managers, and networking outreach.

**Live app:** https://cold-email-gen-aswin.streamlit.app/

---

## 💡 What this app does

This project helps you quickly draft high quality cold emails by combining:

- Your role or goal (internship, full time, networking, etc.)
- The person or company you are reaching out to
- Your skills, experience, or project highlights
- The tone you want (professional, friendly, confident, etc.)

The app sends this information to an LLM via the Groq API and returns a clean, structured email that you can copy, tweak, and send.

Typical use cases:

- Reaching out to data science / ML recruiters  
- Following up after a career fair or event  
- Sending cold intros to hiring managers or founders  

---

## 🧱 Tech Stack

- **Python**
- **Streamlit** for the web UI
- **Groq API** as the LLM backend
- **Requests** for HTTP calls

---

## 📂 Repository structure

```text
cold-email-gen/
├── app.py            # Streamlit app: UI + Groq API call + email generation logic
├── requirements.txt  # Python dependencies
└── README.md         # Project overview and usage instructions
