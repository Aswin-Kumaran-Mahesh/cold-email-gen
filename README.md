# ❄️ Cold Email Generator

LLM powered tool that generates personalized cold emails for recruiters, hiring managers, or networking outreach based on a few simple inputs.

---

## 🚀 What it does

Given:
- Your target (recruiter / hiring manager / founder)
- The role or context (internship, full time, collab, etc.)
- Your skills / projects
- Desired tone

the app calls an LLM through the Groq HTTP API and returns a ready to edit cold email that is:

- Personalized to the company or person
- Focused on your relevant skills
- Structured with a clear call to action

---

## 🧱 Tech Stack

- Python  
- Groq HTTP API (LLM backend)  
- `requests` for HTTP calls  
- Simple script based interface (can be wrapped in a UI later)

---

## 📂 Repository structure

```text
cold-email-gen/
├── app.py            # Main script: prompt template, Groq API call, email generation logic
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation (you are here)
