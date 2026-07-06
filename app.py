import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import pickle
import json
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
from xlsxwriter import Workbook

st.set_page_config(page_title="AI Data Scientist Copilot", layout="wide")

st.title("🤖 AI Data Scientist Copilot")
st.markdown("Upload data, train models, and get AI insights automatically")

# Sidebar
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
task_type = st.sidebar.radio("Task Type", ["Classification", "Regression"])

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.subheader("Data Info")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Shape:**", df.shape)
    with col2:
        st.write("**Columns:**", df.columns.tolist())
    
    target_col = st.selectbox("Select Target Column", df.columns)
    
    if st.button("Train & Compare Models"):
        with st.spinner("Training models..."):
            # Prepare data
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Handle categorical features
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            if y.dtype == 'object':
                le_y = LabelEncoder()
                y = le_y.fit_transform(y.astype(str))
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train models
            if task_type == "Classification":
                models = {
                    'RandomForest': RandomForestClassifier(random_state=42),
                    'LogisticRegression': LogisticRegression(max_iter=1000)
                }
                params = {
                    'RandomForest': {'n_estimators': [50, 100]},
                    'LogisticRegression': {'C': [0.1, 1, 10]}
                }
            else:
                models = {
                    'RandomForest': RandomForestRegressor(random_state=42),
                    'LinearRegression': LinearRegression()
                }
                params = {
                    'RandomForest': {'n_estimators': [50, 100]},
                    'LinearRegression': {}
                }
            
            results = {}
            best_model = None
            best_score = -np.inf
            
            for name, model in models.items():
                # Use cv=2 for small datasets, cv=3 for bigger ones to avoid ValueError
                cv_folds = 2 if len(X_train) < 10 else 3
                grid = GridSearchCV(model, params[name], cv=cv_folds).fit(X_train, y_train)
                pred = grid.predict(X_test)
                
                if task_type == "Classification":
                    score = accuracy_score(y_test, pred)
                    results[name] = {"model": grid.best_estimator_, "score": score}
                else:
                    score = r2_score(y_test, pred)
                    results[name] = {"model": grid.best_estimator_, "score": score}
                
                if score > best_score:
                    best_score = score
                    best_model = grid.best_estimator_
            
            st.success(f"Best Model: {type(best_model).__name__} with Score: {best_score:.4f}")
            
            # SHAP Explanation
            st.subheader("Feature Importance - SHAP")
            explainer = shap.Explainer(best_model, X_train)
            shap_values = explainer(X_test)
            fig = go.Figure()
            # Simplified plot for streamlit
            st.write("SHAP values calculated")
            
            # Download Model
            buf = io.BytesIO()
            pickle.dump(best_model, buf)
            st.download_button("Download Model", buf.getvalue(), "model.pkl")
            
            # Generate PDF Report
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="AI Data Scientist Report", ln=True, align='C')
            pdf.cell(200, 10, txt=f"Best Model: {type(best_model).__name__}", ln=True)
            pdf.cell(200, 10, txt=f"Score: {best_score:.4f}", ln=True)
            pdf_output = pdf.output(dest='S').encode('latin-1')
            st.download_button("Download Report PDF", pdf_output, "report.pdf")
            
            # Export to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Data')
            st.download_button("Download Excel", output.getvalue(), "data.xlsx")

else:
    st.info("Upload a CSV to get started")
