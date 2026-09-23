import io
import json
import hashlib
import os
import re
import sqlite3
from pathlib import Path

import fitz
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CareerLens AI", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DB_PATH = ROOT / "career_lens.db"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
.stApp {
    background:
        radial-gradient(circle at 15% 5%, rgba(124,58,237,.24), transparent 28%),
        radial-gradient(circle at 88% 12%, rgba(14,165,233,.18), transparent 26%),
        linear-gradient(135deg,#070913 0%,#0B1020 45%,#090B14 100%);
}
.block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
.hero {
    padding: 2.2rem 2.3rem;
    border-radius: 28px;
    background: linear-gradient(120deg,rgba(124,58,237,.22),rgba(14,165,233,.14));
    border: 1px solid rgba(255,255,255,.10);
    box-shadow: 0 24px 70px rgba(0,0,0,.28);
    backdrop-filter: blur(14px);
}
.hero h1 {
    font-size: 3.1rem;
    line-height: 1.05;
    margin: 0 0 .7rem 0;
    background: linear-gradient(90deg,#FFFFFF,#C4B5FD,#7DD3FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {font-size:1.05rem;color:#CBD5E1;max-width:850px;margin:0;}
.badge {
    display:inline-block;
    padding:.38rem .7rem;
    border-radius:999px;
    margin:.18rem .2rem .18rem 0;
    background:rgba(124,58,237,.16);
    border:1px solid rgba(196,181,253,.24);
    color:#DDD6FE;
    font-size:.82rem;
}
.glass {
    padding:1.15rem 1.2rem;
    border-radius:20px;
    background:rgba(17,24,39,.72);
    border:1px solid rgba(255,255,255,.08);
    box-shadow:0 12px 38px rgba(0,0,0,.20);
}
.metric-card {
    padding:1.1rem;
    border-radius:18px;
    background:linear-gradient(145deg,rgba(30,41,59,.90),rgba(15,23,42,.78));
    border:1px solid rgba(255,255,255,.08);
    min-height:115px;
}
.metric-title {color:#94A3B8;font-size:.83rem;font-weight:600;}
.metric-value {font-size:1.9rem;font-weight:800;color:#F8FAFC;margin-top:.25rem;}
.metric-sub {color:#A5B4FC;font-size:.78rem;margin-top:.2rem;}
.job-card {
    padding:1.05rem 1.15rem;
    margin:.65rem 0;
    border-radius:18px;
    background:rgba(15,23,42,.82);
    border:1px solid rgba(148,163,184,.13);
}
.score {
    font-size:1.65rem;
    font-weight:800;
    background:linear-gradient(90deg,#C4B5FD,#67E8F9);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.small-muted {color:#94A3B8;font-size:.84rem;}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0B1020,#090B14);
    border-right:1px solid rgba(255,255,255,.07);
}
div.stButton > button {
    border-radius:12px;
    border:1px solid rgba(196,181,253,.32);
    background:linear-gradient(90deg,#6D28D9,#2563EB);
    color:white;
    font-weight:700;
}
div.stButton > button:hover {border-color:#A78BFA;}
[data-testid="stFileUploader"] {
    border:1px dashed rgba(167,139,250,.45);
    border-radius:18px;
    padding:.6rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    classifier = joblib.load(MODEL_DIR / "resume_classifier.joblib")
    vectorizer = joblib.load(MODEL_DIR / "job_vectorizer.joblib")
    job_matrix = joblib.load(MODEL_DIR / "job_matrix.joblib")
    jobs = pd.read_csv(DATA_DIR / "jobs_clean.csv")
    skills = json.loads((DATA_DIR / "skills.json").read_text(encoding="utf-8"))
    return classifier, vectorizer, job_matrix, jobs, skills

def db_connect():
    return sqlite3.connect(DB_PATH)

def init_db():
    with db_connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users(
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses(
            analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            predicted_role TEXT,
            detected_skills TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS applications(
            application_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            job_title TEXT NOT NULL,
            company TEXT,
            status TEXT NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()

def password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 120000).hex()

def register_user(name, email, password):
    salt = os.urandom(16).hex()
    secure_hash = password_hash(password, salt)
    try:
        with db_connect() as conn:
            conn.execute(
                "INSERT INTO users(name,email,password_hash,salt) VALUES(?,?,?,?)",
                (name.strip(), email.strip().lower(), secure_hash, salt)
            )
            conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."

def login_user(email, password):
    with db_connect() as conn:
        row = conn.execute(
            "SELECT user_id,name,email,password_hash,salt FROM users WHERE email=?",
            (email.strip().lower(),)
        ).fetchone()
    if row and password_hash(password, row[4]) == row[3]:
        return {"user_id": row[0], "name": row[1], "email": row[2]}
    return None

def extract_text(uploaded_file):
    content = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".pdf":
        doc = fitz.open(stream=content, filetype="pdf")
        return "\n".join(page.get_text("text") for page in doc)
    if suffix == ".docx":
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    if suffix == ".txt":
        return content.decode("utf-8", errors="ignore")
    return ""

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def detect_skills(text, skills):
    text_clean = " " + clean_text(text) + " "
    found = []
    for skill in skills:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, text_clean):
            found.append(skill)
    return sorted(set(found), key=str.lower)

def job_recommendations(resume_text, resume_skills, jobs, vectorizer, matrix):
    resume_vector = vectorizer.transform([clean_text(resume_text)])
    semantic = cosine_similarity(resume_vector, matrix).ravel()
    resume_skill_set = {s.lower() for s in resume_skills}
    skill_scores = []
    matched = []
    missing = []
    for value in jobs["detected_skills"].fillna(""):
        job_skills = [x.strip() for x in value.split("|") if x.strip()]
        job_set = {x.lower() for x in job_skills}
        overlap = resume_skill_set & job_set
        score = len(overlap) / max(len(job_set), 1)
        skill_scores.append(score)
        matched.append(sorted(overlap))
        missing.append(sorted(job_set - resume_skill_set))
    result = jobs.copy()
    result["semantic_score"] = semantic
    result["skill_score"] = np.array(skill_scores)
    result["match_score"] = (0.72 * result["semantic_score"] + 0.28 * result["skill_score"]) * 100
    result["matched_skills"] = matched
    result["missing_skills"] = missing
    return result.sort_values("match_score", ascending=False).reset_index(drop=True)

def save_analysis(user_id, role, skills):
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO analyses(user_id,predicted_role,detected_skills) VALUES(?,?,?)",
            (user_id, role, " | ".join(skills))
        )
        conn.commit()

init_db()

try:
    classifier, vectorizer, job_matrix, jobs, skills = load_assets()
except Exception:
    st.error("Project assets are missing. Run the complete notebook first, then restart this app.")
    st.stop()

if "user" not in st.session_state:
    st.session_state.user = None
if "guest" not in st.session_state:
    st.session_state.guest = False

with st.sidebar:
    st.markdown("## ✨ CareerLens AI")
    st.caption("Your intelligent career navigator")
    if st.session_state.user:
        st.success(f"Signed in as {st.session_state.user['name']}")
    elif st.session_state.guest:
        st.info("Guest mode")
    page = st.radio(
        "Navigate",
        ["Home", "Analyse Resume", "Job Explorer", "Applications", "Insights", "Methodology"],
        label_visibility="collapsed"
    )
    st.divider()
    if st.session_state.user or st.session_state.guest:
        if st.button("Sign out", use_container_width=True):
            st.session_state.user = None
            st.session_state.guest = False
            st.rerun()

if not st.session_state.user and not st.session_state.guest:
    st.markdown("""
    <div class="hero">
        <span class="badge">NLP</span><span class="badge">Machine Learning</span><span class="badge">Skill Intelligence</span>
        <h1>Turn your resume into a career strategy.</h1>
        <p>CareerLens AI predicts suitable career categories, matches your resume with jobs, identifies missing skills and turns the result into a clear action plan.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    login_tab, register_tab, guest_tab = st.tabs(["Sign in", "Create account", "Explore as guest"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            user = login_user(email, password)
            if user:
                st.session_state.user = user
                st.rerun()
            st.error("Invalid email or password.")
    with register_tab:
        with st.form("register_form"):
            name = st.text_input("Full name")
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            if len(name.strip()) < 2 or "@" not in email or len(password) < 6:
                st.error("Enter a valid name, email and password with at least 6 characters.")
            else:
                ok, message = register_user(name, email, password)
                if ok:
                    st.success(message)
                else:
                    st.error(message)
    with guest_tab:
        st.write("Open the full application without creating an account.")
        if st.button("Continue as guest", use_container_width=True):
            st.session_state.guest = True
            st.rerun()
    st.stop()

if page == "Home":
    st.markdown("""
    <div class="hero">
        <span class="badge">AI Resume Intelligence</span><span class="badge">Explainable Matching</span><span class="badge">Python + ML</span>
        <h1>Find roles that fit. See exactly what is missing.</h1>
        <p>Upload a resume and receive role prediction, job-match scores, matched skills, missing skills and a focused improvement path in one place.</p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    values = [
        ("Job records", f"{len(jobs):,}", "searchable recommendation pool"),
        ("Job categories", f"{jobs['title'].nunique():,}", "different role labels"),
        ("Skill vocabulary", f"{len(skills):,}", "technical and business skills"),
        ("Matching engine", "Hybrid", "TF-IDF + skill coverage")
    ]
    for col, item in zip([c1, c2, c3, c4], values):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-title">{item[0]}</div><div class="metric-value">{item[1]}</div><div class="metric-sub">{item[2]}</div></div>',
                unsafe_allow_html=True
            )
    st.write("")
    left, right = st.columns([1.1, .9])
    with left:
        st.markdown("### What CareerLens does")
        st.markdown("""
        <div class="glass">
        <b>1.</b> Reads PDF, DOCX or TXT resumes<br><br>
        <b>2.</b> Predicts the most suitable resume category using machine learning<br><br>
        <b>3.</b> Detects relevant skills using NLP-style text matching<br><br>
        <b>4.</b> Ranks jobs using TF-IDF similarity and skill coverage<br><br>
        <b>5.</b> Explains the recommendation through matched and missing skills
        </div>
        """, unsafe_allow_html=True)
    with right:
        role_counts = jobs["title"].value_counts().head(10).reset_index()
        role_counts.columns = ["Role", "Jobs"]
        fig = px.bar(role_counts, x="Jobs", y="Role", orientation="h", template="plotly_dark", title="Largest Job Groups")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10), yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)

elif page == "Analyse Resume":
    st.markdown("## Resume Intelligence")
    st.caption("Upload one resume. CareerLens will analyse it against the trained classifier and job corpus.")
    uploaded = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
    if uploaded:
        resume_text = extract_text(uploaded)
        if len(resume_text.strip()) < 80:
            st.error("Very little text could be extracted. Try another PDF, DOCX or TXT file.")
        else:
            cleaned = clean_text(resume_text)
            predicted_role = classifier.predict([cleaned])[0]
            probabilities = classifier.predict_proba([cleaned])[0]
            classes = classifier.classes_
            top_idx = np.argsort(probabilities)[::-1][:5]
            top_roles = pd.DataFrame({"Role": classes[top_idx], "Confidence": probabilities[top_idx] * 100})
            resume_skills = detect_skills(resume_text, skills)
            recommendations = job_recommendations(resume_text, resume_skills, jobs, vectorizer, job_matrix)
            best = recommendations.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Predicted role</div><div class="metric-value" style="font-size:1.25rem">{predicted_role}</div><div class="metric-sub">highest classifier probability</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Detected skills</div><div class="metric-value">{len(resume_skills)}</div><div class="metric-sub">recognised from vocabulary</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Best job match</div><div class="metric-value">{best["match_score"]:.1f}%</div><div class="metric-sub">{best["title"]}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Priority gaps</div><div class="metric-value">{len(best["missing_skills"])}</div><div class="metric-sub">skills missing from top match</div></div>', unsafe_allow_html=True)
            st.write("")
            tab1, tab2, tab3 = st.tabs(["Top Matches", "Skill Gap", "Role Prediction"])
            with tab1:
                for _, row in recommendations.head(7).iterrows():
                    matched_text = ", ".join(row["matched_skills"][:8]) if row["matched_skills"] else "No exact skill overlap detected"
                    st.markdown(
                        f'<div class="job-card"><div class="score">{row["match_score"]:.1f}% match</div><h3 style="margin:.15rem 0">{row["title"]}</h3><div class="small-muted">{row["company"]} · {row["location"]}</div><div style="margin-top:.7rem"><b>Matched skills:</b> {matched_text}</div></div>',
                        unsafe_allow_html=True
                    )
            with tab2:
                top = recommendations.iloc[0]
                l, r = st.columns(2)
                with l:
                    st.success("Matched skills")
                    if top["matched_skills"]:
                        st.markdown(" ".join([f'<span class="badge">{x}</span>' for x in top["matched_skills"]]), unsafe_allow_html=True)
                    else:
                        st.write("No exact skill overlap found.")
                with r:
                    st.warning("Skills to strengthen")
                    if top["missing_skills"]:
                        st.markdown(" ".join([f'<span class="badge">{x}</span>' for x in top["missing_skills"][:18]]), unsafe_allow_html=True)
                    else:
                        st.write("No skill gaps were detected for the top recommendation.")
                gap_data = pd.DataFrame({"Area": ["Matched", "Missing"], "Count": [len(top["matched_skills"]), len(top["missing_skills"])]})
                fig = px.bar(gap_data, x="Area", y="Count", template="plotly_dark", title=f"Skill Coverage for {top['title']}")
                st.plotly_chart(fig, use_container_width=True)
            with tab3:
                fig = px.bar(top_roles.sort_values("Confidence"), x="Confidence", y="Role", orientation="h", template="plotly_dark", title="Top Predicted Career Categories")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(top_roles.style.format({"Confidence":"{:.2f}%"}), use_container_width=True, hide_index=True)
            if st.session_state.user:
                save_analysis(st.session_state.user["user_id"], predicted_role, resume_skills)

elif page == "Job Explorer":
    st.markdown("## Job Explorer")
    c1, c2 = st.columns([1.4, .6])
    with c1:
        query = st.text_input("Search job title, skill or description", placeholder="Python, Data Analyst, SQL...")
    with c2:
        top_n = st.selectbox("Rows", [20, 50, 100], index=0)
    shown = jobs.copy()
    if query.strip():
        q = query.lower().strip()
        mask = (
            shown["title"].fillna("").str.lower().str.contains(q, regex=False) |
            shown["description"].fillna("").str.lower().str.contains(q, regex=False) |
            shown["detected_skills"].fillna("").str.lower().str.contains(q, regex=False)
        )
        shown = shown[mask]
    st.metric("Jobs found", f"{len(shown):,}")
    st.dataframe(shown[["title", "company", "location", "detected_skills"]].head(top_n), use_container_width=True, hide_index=True)

elif page == "Applications":
    st.markdown("## Application Tracker")
    if not st.session_state.user:
        st.info("Create an account and sign in to keep application records linked to your profile.")
    else:
        user_id = st.session_state.user["user_id"]
        with st.form("application_form"):
            title = st.text_input("Job title")
            company = st.text_input("Company")
            status = st.selectbox("Status", ["Saved", "Applied", "Assessment", "Interview", "Offer", "Rejected"])
            notes = st.text_area("Notes")
            submitted = st.form_submit_button("Add application", use_container_width=True)
        if submitted and title.strip():
            with db_connect() as conn:
                conn.execute(
                    "INSERT INTO applications(user_id,job_title,company,status,notes) VALUES(?,?,?,?,?)",
                    (user_id, title.strip(), company.strip(), status, notes.strip())
                )
                conn.commit()
            st.success("Application saved.")
        with db_connect() as conn:
            apps = pd.read_sql_query(
                "SELECT application_id,job_title,company,status,notes,created_at FROM applications WHERE user_id=? ORDER BY application_id DESC",
                conn,
                params=(user_id,)
            )
        if len(apps):
            status_counts = apps["status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig = px.pie(status_counts, names="Status", values="Count", hole=.62, template="plotly_dark", title="Application Pipeline")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(apps, use_container_width=True, hide_index=True)
        else:
            st.info("No applications have been added yet.")

elif page == "Insights":
    st.markdown("## Job Market Insights")
    col1, col2 = st.columns(2)
    with col1:
        role_counts = jobs["title"].value_counts().head(15).reset_index()
        role_counts.columns = ["Role", "Count"]
        fig = px.bar(role_counts.sort_values("Count"), x="Count", y="Role", orientation="h", template="plotly_dark", title="Most Frequent Job Roles")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        exploded = jobs["detected_skills"].fillna("").str.split("|").explode().str.strip()
        exploded = exploded[exploded.ne("")]
        skill_counts = exploded.value_counts().head(15).reset_index()
        skill_counts.columns = ["Skill", "Count"]
        fig = px.bar(skill_counts.sort_values("Count"), x="Count", y="Skill", orientation="h", template="plotly_dark", title="Most Requested Skills")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jobs", f"{len(jobs):,}")
    c2.metric("Unique titles", f"{jobs['title'].nunique():,}")
    c3.metric("Unique companies", f"{jobs['company'].nunique():,}")

elif page == "Methodology":
    st.markdown("## Methodology")
    st.markdown("""
    <div class="glass">
    <h3>1. Resume Parsing</h3>
    PDF files are read with PyMuPDF, DOCX files with python-docx and TXT files directly.<br><br>
    <h3>2. NLP Pre-processing</h3>
    Resume and job text is normalised, cleaned and represented using TF-IDF n-gram features.<br><br>
    <h3>3. Machine Learning Classification</h3>
    The classifier predicts the most suitable resume category from labelled training resumes.<br><br>
    <h3>4. Hybrid Matching</h3>
    Each resume receives a recommendation score combining TF-IDF cosine similarity and explicit skill coverage.<br><br>
    <h3>5. Explainable Skill Gap</h3>
    The recommendation shows both matched skills and skills present in the job profile but not detected in the resume.
    </div>
    """, unsafe_allow_html=True)
