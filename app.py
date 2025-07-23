import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('encoder_scaler.pkl')
REQUIRED_FEATURES = joblib.load('required_features.pkl')

# Preprocessing function
def preprocess_input(df):
    df_encoded = pd.get_dummies(df)

    # Add missing required features with 0
    for col in REQUIRED_FEATURES:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    return df_encoded[REQUIRED_FEATURES]

# Streamlit UI
st.set_page_config(page_title="Employee Salary Predictor", page_icon="🧑‍💻")
st.title("🧑‍💼 Employee Salary Prediction App")

tab1, tab2 = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction (CSV)"])

# --- Tab 1: Single Entry ---
with tab1:
    st.write("Enter the details to predict salary for a single employee.")

    with st.form("prediction_form"):
        experience = st.slider("📅 Years of Experience", 0, 30, 1)
        education = st.selectbox("🎓 Education Level", ['Bachelors', 'Masters', 'PhD'])
        role = st.selectbox("🧰 Job Role", ['Developer', 'Manager', 'Analyst'])
        location = st.selectbox("📍 Location", ['Hyderabad', 'Bangalore', 'Mumbai', 'Pune', 'Delhi'])
        skills = st.selectbox("💻 Skill", ['Python', 'Excel', 'SQL', 'Java'])
        department = st.selectbox("🏢 Department", ['IT', 'Finance', 'HR', 'Marketing'])
        joblevel = st.selectbox("📈 Job Level", ['Junior', 'Mid', 'Senior'])
        certification = st.selectbox("📜 Certifications", ['Yes', 'No'])
        remotework = st.selectbox("🏠 Remote Work", ['Yes', 'No'])

        submitted = st.form_submit_button("🔎 Predict Salary")

    if submitted:
        input_df = pd.DataFrame({
            'Experience': [experience],
            'Education': [education],
            'Role': [role],
            'Location': [location],
            'Skills': [skills],
            'Department': [department],
            'JobLevel': [joblevel],
            'Certifications': [certification],
            'RemoteWork': [remotework]
        })

        processed = preprocess_input(input_df)
        scaled = scaler.transform(processed)
        prediction = model.predict(scaled)[0]
        st.success(f"💰 Estimated Salary: ₹ {int(prediction):,}")

# --- Tab 2: Batch Upload ---
with tab2:
    st.write("Upload a CSV file to predict salaries for multiple employees.")
    st.markdown("📌 **You may upload partial data (even one column), but `Experience` is recommended.**")

    st.download_button("📥 Download Sample CSV",
        data="""Experience,Education,Role,Location,Skills,Department,JobLevel,Certifications,RemoteWork
3,Bachelors,Developer,Bangalore,Python,IT,Junior,Yes,No
5,Masters,Manager,Mumbai,Excel,Finance,Senior,No,Yes
7,PhD,Analyst,Hyderabad,SQL,Marketing,Junior,Yes,No""",
        file_name="sample_employees.csv")

    uploaded_file = st.file_uploader("📤 Upload Your CSV File", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            expected_cols = ['Experience', 'Education', 'Role', 'Location', 'Skills',
                             'Department', 'JobLevel', 'Certifications', 'RemoteWork']

            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 'Unknown' if col != 'Experience' else 0

            processed = preprocess_input(df)
            scaled = scaler.transform(processed)
            predictions = model.predict(scaled)
            df['Predicted Salary'] = predictions.astype(int)

            st.success("✅ Salary prediction completed successfully!")
            st.download_button("📩 Download Predicted Results", df.to_csv(index=False).encode(), "predicted_salaries.csv", "text/csv")
            st.markdown("### 📊 Prediction Results")
            st.dataframe(df)

        except Exception as e:
            st.error(f"❌ Error processing the uploaded file: {e}")
