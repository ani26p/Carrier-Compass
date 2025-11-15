#!/usr/bin/env python3
from flask import Flask, request, jsonify, send_file, render_template_string
import os, io, sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
from fpdf import FPDF
from datetime import datetime

APP_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(APP_DIR, 'Synthetic_job_dataset.csv')

# ---------------- Train All Models --------------------
def train_all_models():

    print("\n===== Training Job & Company Models =====", flush=True)

    if not os.path.exists(DATA_PATH):
        print(f"Dataset missing: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    required_cols = ["Skills", "Past Job", "Past Company", "Current Job", "Current Company", "Experience (Years)"]
    for c in required_cols:
        if c not in df.columns:
            print(f"Missing column: {c}")
            sys.exit(1)

    df = df.fillna("")

    df["Experience (Years)"] = pd.to_numeric(df["Experience (Years)"], errors="coerce").fillna(0)

    df["combined_text"] = (
        df["Skills"].str.lower() + " | " +
        df["Past Job"].str.lower() + " | " +
        df["Past Company"].str.lower() + " | exp:" +
        df["Experience (Years)"].astype(int).astype(str)
    )

    X = df["combined_text"]
    y_job = df["Current Job"].astype(str)
    y_comp = df["Current Company"].astype(str)

    le_job = LabelEncoder()
    le_comp = LabelEncoder()
    y_job_enc = le_job.fit_transform(y_job)
    y_comp_enc = le_comp.fit_transform(y_comp)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=1, sublinear_tf=True)
    X_vec = vectorizer.fit_transform(X)

    models = {
        "logistic": LogisticRegression(max_iter=5000),
        "svm": SVC(kernel="linear", probability=True),
        "random_forest": RandomForestClassifier(n_estimators=250),
        "decision_tree": DecisionTreeClassifier(max_depth=50),
    }

    job_models, comp_models, scores = {}, {}, {}

    for name, mdl in models.items():
        try:
            j = type(mdl)()
            j.fit(X_vec, y_job_enc)
            ja = accuracy_score(y_job_enc, j.predict(X_vec))

            c = type(mdl)()
            c.fit(X_vec, y_comp_enc)
            ca = accuracy_score(y_comp_enc, c.predict(X_vec))

            job_models[name] = j
            comp_models[name] = c
            scores[name] = (ja + ca) / 2

            print(f"{name} → job={ja:.3f}, comp={ca:.3f}")
        except Exception as e:
            print(f"Skipping {name}: {e}")

    best = max(scores, key=scores.get)
    print(f"\nBEST MODEL: {best.upper()} (Accuracy={scores[best]:.3f})")

    return vectorizer, le_job, le_comp, job_models, comp_models, best, scores[best]


vectorizer, le_job, le_comp, JOB_MODELS, COMP_MODELS, BEST_MODEL, BEST_ACC = train_all_models()

app = Flask(__name__)

# ---------------------- FRONTEND HTML -------------------------
HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Carrier Compass</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">

  <style>
    body {
      background: url('https://images.unsplash.com/photo-1521791136064-7986c2920216?auto=format&fit=crop&w=1920&q=80')
                  no-repeat center center fixed;
      background-size: cover;
      font-family: Inter, system-ui;
      backdrop-filter: blur(1px);
    }
    .card {
      border-radius: 16px;
      box-shadow: 0 8px 30px rgba(20,20,50,0.25);
      background: rgba(255,255,255,0.86);
    }
    .pill {
      display:inline-block;
      padding:6px 10px;
      border-radius:999px;
      background:#eef2ff;
      margin:3px;
      font-weight:600;
    }
    .pred-box {
      min-height:72px;
      display:flex;
      align-items:center;
      justify-content:center;
      font-weight:600;
    }
    .title-main {
      text-align:center;
      font-size:90px;
      font-weight:800;
      color:#0d6efd;
      margin-bottom:10px;
    }
    .subtitle {
      text-align:center;
      font-size:28px;
      margin-bottom:20px;
      color:#444;
    }
  </style>
</head>

<body class="p-4">

<div class="container">
  <div class="row mb-4">
    <div class="col-md-8 mx-auto">

      <div class="card p-4">

        <h3 class="title-main">Carrier Compass</h3>
        <p class="subtitle">Job & Company Recommender</p>

        <form id="inputForm" onsubmit="return false;">

          <div class="row g-2">
            <div class="col-md-6"><input id="name" class="form-control" placeholder="Full name"></div>
            <div class="col-md-3"><input id="age" type="number" class="form-control" placeholder="Age"></div>
            <div class="col-md-3"><input id="experience" type="number" class="form-control" placeholder="Experience (years)"></div>
          </div>

          <div class="row g-2 mt-2">
            <div class="col-md-6"><input id="email" type="email" class="form-control" placeholder="Email"></div>
            <div class="col-md-6"><input id="phone" class="form-control" placeholder="Phone number"></div>
          </div>

          <div class="row g-2 mt-2">
            <div class="col-12"><textarea id="skills" class="form-control" placeholder="Skills (comma separated)"></textarea></div>
          </div>

          <div class="row g-2 mt-2">
            <div class="col-md-6"><input id="pastJob" class="form-control" placeholder="Past job role"></div>
            <div class="col-md-6"><input id="pastCompany" class="form-control" placeholder="Past company"></div>
          </div>

          <div class="row g-2 mt-3 align-items-center">

            <!-- CENTERED + WIDER STATUS BOX -->
            <div class="col-md-7 d-flex justify-content-center">
              <input id="modelStatus" class="form-control text-center"
                     style="font-weight:bold; color:#0d6efd; width:100%; max-width:500px;"
                     value="CARRIER COMPASS" disabled>
            </div>

            <!-- Buttons -->
            <div class="col-md-5 text-end">
              <button class="btn btn-primary" id="predictBtn">Predict</button>
              <button class="btn btn-outline-secondary" id="genResumeBtn">Generate Resume</button>
            </div>
          </div>

        </form>

        <hr>

        <div class="row">
          <div class="col-md-6">
            <div class="card p-3">
              <h6>Job role suggestions</h6>
              <div id="jobPred" class="pred-box">—</div>
            </div>
          </div>
          <div class="col-md-6">
            <div class="card p-3">
              <h6>Company suggestions</h6>
              <div id="compPred" class="pred-box">—</div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>
</div>

<script>

let fullJobList = [];
let fullCompList = [];
let displayedJobs = [];
let displayedComps = [];

// Pick 2 random
function pickTwoRandom(arr){
  if(arr.length <= 1) return arr;
  let a = Math.floor(Math.random() * arr.length);
  let b = Math.floor(Math.random() * arr.length);
  while(b === a) b = Math.floor(Math.random() * arr.length);
  return [arr[a], arr[b]];
}

document.getElementById("predictBtn").addEventListener("click", async (e)=>{
  e.preventDefault();

  // ⛔ Show error if skills empty
  if (skills.value.trim() === "") {
    alert("Please enter your skills before predicting!");
    return;
  }

  const modelBox = document.getElementById("modelStatus");

  // Show processing message
  modelBox.value = "Processing... Please wait";
  document.getElementById("predictBtn").disabled = true;

  // Wait 3 seconds
  await new Promise(r => setTimeout(r, 1500));

  // Prepare payload
  const payload = {
    skills: skills.value,
    pastJob: pastJob.value,
    pastCompany: pastCompany.value,
    experience: experience.value
  };

  const res = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });

  const data = await res.json();

  fullJobList = data.job_roles;
  fullCompList = data.companies;

  displayedJobs = pickTwoRandom(fullJobList);
  displayedComps = pickTwoRandom(fullCompList);

  jobPred.innerHTML = displayedJobs.map(j => `<span class='pill'>${j}</span>`).join("");
  compPred.innerHTML = displayedComps.map(j => `<span class='pill'>${j}</span>`).join("");

  // Show model info again
  modelBox.value = "Best Model: {{model}} (Accuracy: {{acc}}%)";

  document.getElementById("predictBtn").disabled = false;
});

// Resume button popup + generate
document.getElementById("genResumeBtn").addEventListener("click", async (e)=>{
  e.preventDefault();

  alert("Your resume is being generated...");

  const payload = {
    name: name.value,
    email: email.value,
    phone: phone.value,
    skills: skills.value,
    pastJob: pastJob.value,
    pastCompany: pastCompany.value,
    experience: experience.value,
    job_roles: displayedJobs,
    companies: displayedComps
  };

  const res = await fetch("/generate_resume", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "resume.pdf";
  a.click();
});
</script>

</body>
</html>
"""

# ---------------------- BACKEND -------------------------
@app.route("/")
def home():
    return render_template_string(HTML, model=BEST_MODEL.upper(), acc=round(BEST_ACC*100, 2))


def top_k_by_proba(model, X_vec, enc, k=6):
    try:
        probs = model.predict_proba(X_vec)[0]
    except:
        scores = model.decision_function(X_vec)[0]
        scores = np.array(scores)
        if scores.ndim == 0:
            scores = np.array([-scores, scores])
        probs = scores

    probs = np.asarray(probs).ravel()
    idx = np.argsort(-probs)[:k]

    try:
        return list(enc.inverse_transform(idx))
    except:
        return list(np.array(enc.classes_)[idx])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    text = " | ".join([
        data.get("skills",""),
        data.get("pastJob",""),
        data.get("pastCompany",""),
        "exp:" + str(data.get("experience",0))
    ])

    vec = vectorizer.transform([text])

    job = top_k_by_proba(JOB_MODELS[BEST_MODEL], vec, le_job, k=6)
    comp = top_k_by_proba(COMP_MODELS[BEST_MODEL], vec, le_comp, k=6)

    return jsonify({"job_roles": job, "companies": comp})


@app.route("/generate_resume", methods=["POST"])
def generate_resume():
    data = request.get_json()

    name = data.get("name","")
    email = data.get("email","")
    phone = data.get("phone","")
    skills = data.get("skills","")
    past_job = data.get("pastJob","")
    past_company = data.get("pastCompany","")

    job_suggestions = ", ".join(data.get("job_roles",[]))
    company_suggestions = ", ".join(data.get("companies",[]))

    line = "-" * 120

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0,8,line)
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,name.upper(),ln=True,align="C")
    pdf.set_font("Arial",size=12)
    pdf.multi_cell(0,8,line)

    pdf.cell(0,8,f"Email - {email}",ln=True,align="R")
    pdf.cell(0,8,f"Mobile - {phone}",ln=True,align="R")
    pdf.multi_cell(0,8,line)
    pdf.ln(4)

    pdf.cell(0,8,f"Date - {datetime.now().strftime('%d-%m-%Y')}",ln=True)
    pdf.ln(4)

    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,"Skills",ln=True)
    pdf.set_font("Arial",size=12)
    pdf.multi_cell(0,8,f"- {skills}")
    pdf.ln(4)

    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,"Past Experience",ln=True)
    pdf.set_font("Arial",size=12)
    pdf.cell(0,8,f"Role: {past_job}",ln=True)
    pdf.cell(0,8,f"Companies: {past_company}",ln=True)

    pdf.multi_cell(0,8,line)
    pdf.ln(4)

    pdf.set_font("Arial","B",12)
    pdf.cell(0,8,"Carrier-Compass Suggestions -",ln=True)
    pdf.set_font("Arial",size=12)
    pdf.multi_cell(0,8,f"Suggested Roles: {job_suggestions}")
    pdf.multi_cell(0,8,f"Suggested Companies: {company_suggestions}")

    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    return send_file(io.BytesIO(pdf_bytes),
                     mimetype="application/pdf",
                     as_attachment=True,
                     download_name="resume.pdf")


if __name__ == "__main__":
    print("Running at http://127.0.0.1:5000")
    app.run(debug=True)
