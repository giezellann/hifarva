# Campus Facility Reservation Monitoring: Streamlit Deployment

Course: Systems Analysis and Design (BS Information Systems), Section A.
Deploys the **Midterm Laboratory Examination** (cancellation / no-show prediction) as a decision-support web app.

## What it does
| Page | Module | Purpose |
|---|---|---|
| Home | overview | model summary, target, features, limitations |
| Dataset | 1 | CSV upload, schema check, preview, null/duplicate summary |
| Data Analysis | 2 | descriptive stats, class distribution, 5 interactive Plotly charts |
| Prediction | 3 | single / batch prediction with the saved pipeline, metrics, limitations |
| Dashboard | 4 | KPIs, facility risk and utilization, risk queue, recommended follow-up, human decision log |
| System Design | 5 | 5 FRs, 3 NFRs, actors, use case diagram, decision workflow |

**Target:** `Reservation_Outcome` (1 = Not Completed: Cancelled or No-Show, 0 = Completed)
**Features (available at booking time only):** User_Category, Facility_Type, Lead_Time_Days, Department, Time_Slot, Participants
**Leakage removed:** Usage_Status, Actual_Attendance, Attendance_Ratio
**Model:** DecisionTreeClassifier(max_depth=4, random_state=42), 80/20 stratified split. Results equal the midterm:
confusion matrix [[38,12],[25,5]], accuracy 0.5375, precision 0.2941, recall 0.1667, F1 0.2128 (398 records after cleaning).

## Repository layout
```
app.py
requirements.txt
README.md
modules/ data_processing.py exploratory_analysis.py prediction.py dashboard.py system_design.py ui.py
models/  trained_model.joblib  model_metrics.json
assets/  use_case_diagram.png  decision_workflow.png
data/    facility_reservations.csv (synthetic, from midterm seed 42)  facility_reservations_cleaned.csv
notebooks/midterm_examination.ipynb   (original midterm notebook)
scripts/ train_model.py (rebuilds the model)  make_diagrams.py
tests/   test_logic.py
```
No passwords, secrets or real student records are included. The data is synthetic.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Optional: `python scripts/train_model.py` rebuilds `models/trained_model.joblib` (needed only if you change the data/model).
`python tests/test_logic.py` runs the consistency checks. `pip install matplotlib` first if you want to rerun `scripts/make_diagrams.py`.

## Deploy (Streamlit Community Cloud)
1. Create a GitHub repository and push everything in this folder (`git init`, `git add .`, `git commit`, `git push`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose the repository and branch, set the entry point to `app.py`, then **Deploy**.
4. Open the app URL. If the build fails, open **Manage app > Logs**.
5. Submit the GitHub URL and the deployed Streamlit URL.

Note: the saved model was built with scikit-learn 1.9.1 (pinned in requirements.txt). If a different version is installed
and the file cannot be loaded, the app rebuilds the same pipeline once from the bundled dataset and shows a notice.

## Test checklist (record screenshots for each)
- [ ] Deployed URL opens without error; all six pages work
- [ ] Upload `data/facility_reservations.csv` -> accepted; upload a CSV missing `Participants` -> clear error; upload an empty CSV -> clear error
- [ ] Prediction: valid input works; negative lead time / zero participants shows an error
- [ ] Five sample predictions (e.g. Student/Lecture Hall/10 days/College of Eng/Morning/80 participants) match `pipe.predict_proba`
- [ ] Charts react to the facility filter
- [ ] Dashboard: role "Facility Administrator" can save a decision; "Requester" cannot see scores
- [ ] Model version, prediction timestamp, limitations and human review notice are visible

## Limitations
- Synthetic data with mostly random patterns; the model is weak (recall 16.7%, accuracy 53.75%), so it misses most real cancellations / no-shows.
- Dashboard scores on the midterm records are in-sample (model trained on them) and illustrate the workflow, not future accuracy.
- Decision log lives in the session only (download it as CSV); a production system would store it in a database.
- Predictions are decision support. They must not automatically reject a reservation or deny student services.
