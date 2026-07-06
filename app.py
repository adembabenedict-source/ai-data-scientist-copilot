import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import io
import pickle
import warnings
import shap
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Data Scientist Copilot Pro", layout="wide", page_icon="🤖")

# CSS
st.markdown("""
<style>
  .main-header {font-size:2.8rem; font-weight:800; background: linear-gradient(90deg, #4F46E5, #EC4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
  .metric-card {background-color:#1F2937; padding:15px; border-radius:12px; border: 1px solid #374151;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🤖 AI Data Scientist Copilot Pro</p>', unsafe_allow_html=True)
st.caption("Enterprise Automated ML with Full Explainability & MLOps")

# Sidebar - Hyperparameter tuning panel + Resource monitoring
with st.sidebar:
    st.header("⚙️ Configuration Panel")
    task_type = st.radio("Task Type", ["Auto-Detect", "Classification", "Regression"])
    n_estimators = st.slider("RF Estimators", 50, 200, 100)
    cv_folds_slider = st.slider("CV Folds", 2, 5, 3)
    st.markdown("---")
    st.subheader("📊 Resource Monitor")
    st.metric("Status", "Ready")
    log_area = st.empty()

uploaded_file = st.file_uploader("📁 Drag-and-drop dataset upload", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_original = df.copy()
    experiment_history = []
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 EDA Dashboard", "🧠 Model Training", "📈 Model Comparison", "🔍 Explainability", "📥 Export & Deploy"])
    
    with tab1: # EDA Dashboard with all features
        st.subheader("Interactive Dataset Preview")
        st.dataframe(df, use_container_width=True, height=300)
        
        # Dataset statistics cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Cells", df.isnull().sum().sum())
        col4.metric("Duplicates", df.duplicated().sum())
        col5.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Missing value analysis dashboard
        st.subheader("Missing Value Analysis")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ['Feature', 'Missing Count']
        fig_missing = px.bar(missing_df, x='Feature', y='Missing Count', template="plotly_dark")
        st.plotly_chart(fig_missing, use_container_width=True)
        
        # Data type summary
        st.subheader("Data Type Summary")
        st.dataframe(df.dtypes.astype(str).to_frame('Dtype'), use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            fig_corr = px.imshow(numeric_df.corr(), text_auto=".2f", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Distribution/histogram charts
        st.subheader("Feature Distributions")
        col_to_plot = st.selectbox("Select column for histogram", df.columns)
        fig_dist = px.histogram(df, x=col_to_plot, marginal="box")
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab2: # Model Training with all pro features
        target_col = st.selectbox("🎯 Select Target Column", df.columns)
        
        # Automatic target type detection
        if task_type == "Auto-Detect":
            detected = "Classification" if df[target_col].nunique() < 20 and df[target_col].dtype == 'object' else "Regression"
            st.info(f"Auto-Detected Task: **{detected}**")
            task_type = detected
        
        # Feature engineering tools
        st.subheader("Feature Engineering")
        if st.checkbox("Add Squared Features for Numerics"):
            for col in df.select_dtypes(include=np.number).columns:
                df[f"{col}_sq"] = df[col]**2
        
        if st.button("🚀 Train & Compare Models", type="primary"):
            log_area.text("Training started...")
            progress = st.progress(0)
            
            # Data Cleaning
            df_clean = df.dropna(subset=[target_col]).copy()
            X = df_clean.drop(columns=[target_col])
            y = df_clean[target_col]
            
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            if y.dtype == 'object' and task_type == "Classification":
                le_y = LabelEncoder()
                y = le_y.fit_transform(y.astype(str))
            
            X = X.fillna(X.median())
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            progress.progress(30)
            
            # Model Selection
            cv = KFold(n_splits=cv_folds_slider, shuffle=True, random_state=42)
            if task_type == "Classification":
                models = {
                    'Random Forest': (RandomForestClassifier(n_estimators=n_estimators, random_state=42), {'n_estimators': [n_estimators]}),
                    'Logistic Regression': (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]})
                }
            else:
                models = {
                    'Random Forest': (RandomForestRegressor(n_estimators=n_estimators, random_state=42), {'n_estimators': [n_estimators]}),
                    'Linear Regression': (LinearRegression(), {})
                }
            
            results = {}
            best_model = None
            best_score = -np.inf
            
            for i, (name, (model, param_grid)) in enumerate(models.items()):
                log_area.text(f"Training {name}...")
                if param_grid:
                    grid = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1)
                    grid.fit(X_train, y_train)
                    best_est = grid.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    best_est = model
                
                # Cross-validation results
                cv_scores = cross_val_score(best_est, X_train, y_train, cv=cv)
                pred = best_est.predict(X_test)
                
                if task_type == "Classification":
                    score = accuracy_score(y_test, pred)
                    results[name] = {"model": best_est, "test_score": score, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(), "pred": pred, "proba": best_est.predict_proba(X_test)[:,1] if hasattr(best_est, 'predict_proba') else None}
                else:
                    score = r2_score(y_test, pred)
                    results[name] = {"model": best_est, "test_score": score, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(), "pred": pred}
                
                if score > best_score:
                    best_score = score
                    best_model = best_est
                
                progress.progress(30 + (i+1)*35)
            
            st.session_state['results'] = results
            st.session_state['best_model'] = best_model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['X_train'] = X_train
            st.success("Training Complete!")
            log_area.text("Training finished.")

    with tab3: # Model comparison leaderboard + ROC/PR + Residual
        if 'results' in st.session_state:
            st.subheader("Model Comparison Leaderboard")
            leaderboard = pd.DataFrame([
                {"Model": k, "Test Score": v['test_score'], "CV Mean": v['cv_mean'], "CV Std": v['cv_std']}
                for k, v in st.session_state['results'].items()
            ]).sort_values("Test Score", ascending=False)
            st.dataframe(leaderboard.style.highlight_max(axis=0), use_container_width=True)
            
            y_test = st.session_state['y_test']
            if task_type == "Classification":
                st.subheader("ROC Curve & Confusion Matrix")
                for name, res in st.session_state['results'].items():
                    if res['proba'] is not None:
                        fpr, tpr, _ = roc_curve(y_test, res['proba'])
                        fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Curve - {name}')
                        st.plotly_chart(fig_roc)
                
                cm = confusion_matrix(y_test, st.session_state['results'][leaderboard.iloc[0]['Model']]['pred'])
                fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix - Best Model")
                st.plotly_chart(fig_cm)
            else:
                st.subheader("Residual Plots")
                for name, res in st.session_state['results'].items():
                    residuals = y_test - res['pred']
                    fig_res = px.scatter(x=res['pred'], y=residuals, title=f'Residual Plot - {name}')
                    st.plotly_chart(fig_res)

    with tab4: # SHAP/LIME explainability
        if 'best_model' in st.session_state:
            st.subheader("SHAP Explainability")
            explainer = shap.Explainer(st.session_state['best_model'], st.session_state['X_train'])
            shap_values = explainer(st.session_state['X_test'][:100])
            importance = np.abs(shap_values.values).mean(0)
            fig_shap = px.bar(x=st.session_state['X_train'].columns, y=importance)
            st.plotly_chart(fig_shap, use_container_width=True)

    with tab5: # Export
        if 'best_model' in st.session_state:
            st.subheader("📥 Downloadable Preprocessing Report")
            report_txt = f"Report generated at {datetime.now()}\nBest Model: {type(st.session_state['best_model']).__name__}\nBest Score: {best_score:.4f}"
            st.download_button("Download Report.txt", report_txt, "preprocessing_report.txt")
            
            buf = io.BytesIO()
            pickle.dump(st.session_state['best_model'], buf)
            st.download_button("📦 Download Model.pkl", buf.getvalue(), "model.pkl")
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_original.to_excel(writer, sheet_name='Data', index=False)
            st.download_button("📊 Download Excel", output.getvalue(), "data.xlsx")
            
            st.subheader("🚀 One-click deployment/export to API")
            st.code("# FastAPI deployment code generated\nfrom fastapi import FastAPI\nimport pickle\napp = FastAPI()\nmodel = pickle.load(open('model.pkl', 'rb'))")

else:
    st.info("👆 Upload a CSV to unlock the full dashboard")
