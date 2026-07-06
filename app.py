import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import io, pickle, json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

st.set_page_config(page_title="AI Data Scientist Copilot Pro", layout="wide")

# === SESSION STATE MANAGEMENT + LOGGING ===
if 'df' not in st.session_state: st.session_state.df = None
if 'model' not in st.session_state: st.session_state.model = None
if 'logs' not in st.session_state: st.session_state.logs = []

def log(msg):
    st.session_state.logs.append(msg)

st.title("🤖 AI Data Scientist Copilot Pro")
st.write("Upload your data and let AI handle EDA, Modeling, and Reporting")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.session_state.df = df
    log(f"Loaded {uploaded_file.name} with shape {df.shape}")

if st.session_state.df is not None:
    df = st.session_state.df
    
    tab1, tab2, tab3 = st.tabs(["📊 EDA", "🤖 Model", "📄 Report"])
    
    with tab1:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        st.subheader("Summary Stats")
        st.write(df.describe())
        
        st.subheader("Correlation Heatmap")
        fig = px.imshow(df.corr(numeric_only=True), text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Train Model")
        target = st.selectbox("Select Target Column", df.columns)
        X = df.drop(columns=[target])
        y = df[target]
        
        # Encode categorical
        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col])
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        model_type = st.selectbox("Model Type", ["RandomForest", "Logistic/Linear"])
        
        if st.button("Train"):
            if model_type == "RandomForest":
                if y.dtype == 'object' or len(y.unique()) < 10:
                    model = RandomForestClassifier()
                else:
                    model = RandomForestRegressor()
            else:
                if y.dtype == 'object' or len(y.unique()) < 10:
                    model = LogisticRegression(max_iter=1000)
                else:
                    model = LinearRegression()
            
            model.fit(X_train, y_train)
            st.session_state.model = model
            preds = model.predict(X_test)
            
            if hasattr(model, 'predict_proba'):
                st.write("Accuracy:", accuracy_score(y_test, preds))
            else:
                st.write("R2 Score:", r2_score(y_test, preds))
            log("Model Trained")
    
    with tab3:
        st.subheader("Download Report")
        st.write("Export your data and results")
        
        # CSV DOWNLOAD INSTEAD OF PDF
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name='report.csv',
            mime='text/csv',
        )
        
        st.write("Logs:")
        for l in st.session_state.logs:
            st.write("-", l)
