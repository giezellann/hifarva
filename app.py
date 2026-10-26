"""Campus Facility Reservation Monitoring: Streamlit deployment of the Midterm Laboratory Examination."""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SAD Predictive System", layout="wide")

from modules import ui  # noqa: E402
from modules.dashboard import render_dashboard_page  # noqa: E402
from modules.data_processing import (REQUIRED_COLUMNS, cached_clean, load_csv_bytes,  # noqa: E402
                                     load_default_data, quality_summary, validate_schema)
from modules.exploratory_analysis import render_eda_page  # noqa: E402
from modules.prediction import load_metrics, load_model, render_prediction_page  # noqa: E402
from modules.system_design import render_system_design_page  # noqa: E402

menu = st.sidebar.selectbox("Module", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
st.sidebar.caption("Section A | Campus Facility Reservation Monitoring")


def active_data():
    """Uploaded (validated) data if present, otherwise the bundled midterm dataset."""
    up = st.session_state.get("uploaded_df")
    if up is not None:
        return up, f"uploaded file ({st.session_state.get('uploaded_name', 'CSV')})"
    return load_default_data(), "bundled midterm dataset (synthetic, 400 rows)"


if menu == "Home":
    m = load_metrics()
    st.title("Campus Facility Reservation Monitoring")
    st.subheader("Predicting cancellation / no-show risk: deployment of the Midterm Laboratory Examination")
    st.write("This app turns the midterm data-mining work into an information-system workflow. A saved Decision Tree "
             "pipeline scores reservations, a dashboard ranks the riskiest bookings, and a **facility administrator** "
             "records the final decision.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Decision Tree (depth 4)")
    c2.metric("Accuracy", f"{m['accuracy']:.4f}")
    c3.metric("Recall", f"{m['recall']:.4f}")
    c4.metric("F1-score", f"{m['f1']:.4f}")
    st.markdown("""
**Modules:** Dataset (upload and validation) | Data Analysis (EDA) | Prediction | Dashboard | System Design

**Target:** `Reservation_Outcome` where 1 = Not Completed (Cancelled or No-Show) and 0 = Completed.
**Predictors (known at booking time):** User_Category, Facility_Type, Lead_Time_Days, Department, Time_Slot, Participants.
Leakage fields (`Usage_Status`, `Actual_Attendance`, `Attendance_Ratio`) are never given to the model.

**Differences from the original midterm problem:** none in data, target, features or results. The deployed pipeline
reproduces the midterm test results. The only addition is the one-hot encoder now lives inside the saved pipeline instead of
`pd.get_dummies`, which gives identical columns.
""")
    st.warning("Limitations: the dataset is synthetic and the model is weak (low recall). Scores are decision support, "
               "not authorization to reject a reservation or deny student services.")
    st.caption(f"Model version {m['model_version']} | trained {m['trained_at']} | scikit-learn {m['sklearn_version']}")

elif menu == "Dataset":
    st.header("Dataset Upload and Validation")
    st.write("Required columns: " + ", ".join(REQUIRED_COLUMNS))
    f = st.file_uploader("Upload CSV", type="csv")
    if f is not None:
        try:
            df = load_csv_bytes(f.getvalue())
        except Exception as exc:
            st.error(f"Could not read this file as a CSV: {exc}")
            df = None
        if df is not None:
            ok, msgs = validate_schema(df)
            for msg in msgs:
                st.error(msg)
            if ok:
                st.success("Schema check passed. This file is now used by the Data Analysis, Prediction and Dashboard modules.")
                st.session_state["uploaded_df"] = df
                st.session_state["uploaded_name"] = f.name
            else:
                st.session_state.pop("uploaded_df", None)
                st.info("The file was not accepted. The bundled midterm dataset is still in use.")
    else:
        if st.session_state.get("uploaded_df") is not None and st.button("Remove uploaded file"):
            st.session_state.pop("uploaded_df", None)
            st.rerun()

    df, source = active_data()
    st.caption(f"Currently using: {source}")
    q = quality_summary(df)
    a, b, c, d = st.columns(4)
    a.metric("Rows", q["rows"])
    b.metric("Columns", q["columns"])
    c.metric("Missing values", q["total_nulls"])
    d.metric("Duplicate Reservation_IDs", q["duplicate_ids"])
    st.subheader("Data preview")
    ui.show_table(df.head(20))
    st.subheader("Null summary")
    if q["total_nulls"]:
        ui.show_table(q["null_counts"].rename("Null_Count").to_frame())
    else:
        st.write("No missing values.")
    st.write(f"Duplicate rows: **{q['duplicate_rows']}**. Duplicate IDs are removed by the midterm cleaning step "
             "(first occurrence kept).")
    with st.expander("Result of the midterm cleaning on this data"):
        try:
            cl = cached_clean(df)
            st.write(f"{len(df)} rows -> {len(cl)} rows after cleaning.")
            ui.show_table(cl.head(20))
        except Exception as exc:
            st.error(f"Cleaning failed: {exc}")

elif menu == "Data Analysis":
    df, source = active_data()
    render_eda_page(df, source)

elif menu == "Prediction":
    render_prediction_page(ui)

elif menu == "Dashboard":
    df, source = active_data()
    render_dashboard_page(df, source)

elif menu == "System Design":
    render_system_design_page()
