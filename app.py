import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import pickle
import warnings
import shap
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from fpdf import FPDF

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Data Scientist Copilot Pro", layout="wide", page_icon="🤖")

# Custom CSS for professional look
st.markdown("""
<style>
   .main-header {font-size:3rem; font-weight:700; color:#4F46E5;}
   .metric-card {background-color:#F3F4F6; padding:20px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🤖 AI Data Scientist Copilot Pro</p>', unsafe_allow_html=True)
st.caption("Enterprise-grade automated ML with explainability and reporting")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    task_type = st.radio("Select Task Type", ["Classification", "Regression"], horizontal=True)
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    st.markdown("---")
    st.info("Upload a CSV with >10 rows for best results")

uploaded_file = st.file_uploader("📁 Upload your Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🧠 Model Training", "📥 Export"])
    
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Rows", df.shape[0])
        with col2: st.metric("Columns", df.shape[1])
        with col3: st.metric("Missing Values", df.isnull().sum().sum())
        with col4: st.metric("Duplicates", df.duplicated().sum())
        
        st.subheader("Data Types")
        st.dataframe(df.dtypes.astype(str), use_container_width=True)

    with tab2:
        target_col = st.selectbox("🎯 Select Target Column", df.columns)
        
        if st.button("🚀 Train & Compare Models", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            
            try:
                # 1. Data Cleaning
                status.text("Step 1/5: Cleaning data...")
                df_clean = df.dropna(subset=[target_col]).copy()
                X = df_clean.drop(columns=[target_col])
                y = df_clean[target_col]
                
                if len(X) < 6:
                    st.error("❌ Dataset too small. Need minimum 6 rows after cleaning.")
                    st.stop()
                
                # Preprocessing Pipeline
                numeric_features = X.select_dtypes(include=[np.number]).columns
                categorical_features = X.select_dtypes(include=['object']).columns
                
                for col in categorical_features:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                
                imputer = SimpleImputer(strategy='median')
                scaler = StandardScaler()
                X[numeric_features] = scaler.fit_transform(imputer.fit_transform(X[numeric_features]))
                
                if y.dtype == 'object' and task_type == "Classification":
                    le_y = LabelEncoder()
                    y = le_y.fit_transform(y.astype(str))
                
                progress.progress(20)
                
                # 2. Train/Test Split
                status.text("Step 2/5: Splitting data...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                progress.progress(40)
                
                # 3. Model Selection with robust CV
                status.text("Step 3/5: Training models...")
                cv_folds = 2 if len(X_train) < 10 else min(5, len(X_train)//2)
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                if task_type == "Classification":
                    models = {
                        'Random Forest': (RandomForestClassifier(random_state=42), {'n_estimators': [100]}),
                        'Logistic Regression': (LogisticRegression(max_iter=1000), {'C': [1]})
                    }
                    metric_name = "Accuracy"
                else:
                    models = {
                        'Random Forest': (RandomForestRegressor(random_state=42), {'n_estimators': [100]}),
                        'Linear Regression': (LinearRegression(), {})
                    }
                    metric_name = "R² Score"
                
                results = {}
                best_model = None
                best_score = -np.inf
                
                for name, (model, param_grid) in models.items():
                    try:
                        if param_grid:
                            grid = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, error_score='raise')
                            grid.fit(X_train, y_train)
                            best_est = grid.best_estimator_
                        else:
                            model.fit(X_train, y_train)
                            best_est = model
                        
                        pred = best_est.predict(X_test)
                        score = accuracy_score(y_test, pred) if task_type == "Classification" else r2_score(y_test, pred)
                        
                        results[name] = {"model": best_est, "score": score}
                        if score > best_score:
                            best_score = score
                            best_model = best_est
                            
                    except Exception as e:
                        st.warning(f"⚠️ {name} skipped: {str(e)[:80]}")
                        continue
                
                progress.progress(80)
                
                if best_model is None:
                    st.error("❌ All models failed. Ensure target is numeric for Regression.")
                    st.stop()
                
                status.text("Step 4/5: Generating insights...")
                
                # 4. Results Display
                st.success(f"✅ Training Complete! Best Model: **{type(best_model).__name__}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"Best {metric_name}", f"{best_score:.4f}")
                with col2:
                    st.metric("Models Compared", len(results))
                
                # Feature Importance
                st.subheader("📈 Feature Importance")
                try:
                    explainer = shap.Explainer(best_model, X_train)
                    shap_values = explainer(X_test[:100])
                    importance = np.abs(shap_values.values).mean(0)
                    fig = px.bar(x=X.columns, y=importance, labels={'x':'Features', 'y':'Mean |SHAP|'})
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Feature importance unavailable for this model")
                
                progress.progress(100)
                status.text("Step 5/5: Done!")
                
                st.session_state['best_model'] = best_model
                st.session_state['df'] = df
                
            except Exception as e:
                st.error(f"Critical Error: {e}")
    
    with tab3:
        if 'best_model' in st.session_state:
            st.subheader("📥 Export Results")
            
            # Download Model
            buf = io.BytesIO()
            pickle.dump(st.session_state['best_model'], buf)
            st.download_button("📦 Download Trained Model", buf.getvalue(), "model.pkl")
            
            # PDF Report
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "AI Data Scientist Report", 0, 1, 'C')
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
            pdf.cell(0, 10, f"Best Model: {type(st.session_state['best_model']).__name__}", 0, 1)
            pdf.cell(0, 10, f"Best Score: {best_score:.4f}", 0, 1)
            pdf_output = pdf.output(dest='S').encode('latin-1')
            st.download_button("📄 Download PDF Report", pdf_output, "report.pdf")
            
            # Excel Export
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
            st.download_button("📊 Download Excel", output.getvalue(), "data.xlsx")
        else:
            st.info("Train a model first to enable exports")

else:
    st.info("👆 Upload a CSV file to begin")
