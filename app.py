import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import io, pickle, warnings, shap, time, psutil, json, os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')
PROJECT_DIR = "projects"
os.makedirs(PROJECT_DIR, exist_ok=True)

# Session state for history
if 'experiment_history' not in st.session_state:
    st.session_state['experiment_history'] = []

st.set_page_config(page_title="AI Data Scientist Copilot Pro", layout="wide", page_icon="🤖")

# Light/Dark mode toggle
theme = st.sidebar.toggle("🌙 Dark Mode", value=True)
theme_css = "background-color:#0E1117; color:white;" if theme else "background-color:white; color:black;"
st.markdown(f"<style>.stApp {{{theme_css}}}</style>", unsafe_allow_html=True)

st.markdown('<h1 style="font-size:2.8rem; font-weight:800;">🤖 AI Data Scientist Copilot Pro</h1>', unsafe_allow_html=True)

# Sidebar - Auth + Hyperparameter + Resource Monitor + Notifications
with st.sidebar:
    st.header("👤 User Profile")
    username = st.text_input("Username", "guest")
    st.markdown("---")
    
    st.header("⚙️ Hyperparameter Tuning Panel")
    task_type = st.radio("Task Type", ["Auto-Detect", "Classification", "Regression"])
    n_estimators = st.slider("RF Estimators", 50, 300, 100)
    cv_folds = st.slider("CV Folds", 2, 5, 3)
    
    st.markdown("---")
    st.header("📊 Live Resource Monitor")
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    st.progress(cpu/100, text=f"CPU: {cpu}%")
    st.progress(ram/100, text=f"RAM: {ram}%")
    
    st.markdown("---")
    st.header("🔔 Notification Center")
    notif = st.empty()

uploaded_file = st.file_uploader("📁 Drag-and-drop dataset upload", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Guided onboarding/tutorial
    with st.expander("📖 Guided Onboarding - Click to see steps", expanded=False):
        st.markdown("1. Upload CSV 2. Select Target 3. Click Train 4. View Results 5. Export Model")
    
    tabs = st.tabs(["📊 EDA", "🧠 Training", "🏆 Leaderboard", "🔍 Explain", "📁 Projects", "📥 Export"])
    
    with tabs[0]: # EDA
        st.subheader("Interactive Dataset Preview Table")
        st.dataframe(df, use_container_width=True, height=300)
        
        # Dataset statistics cards
        cols = st.columns(5)
        cols[0].metric("Rows", df.shape[0])
        cols[1].metric("Cols", df.shape[1])
        cols[2].metric("Missing", df.isnull().sum().sum())
        cols[3].metric("Duplicates", df.duplicated().sum())
        cols[4].metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Missing value analysis
        st.subheader("Missing Value Analysis")
        st.bar_chart(df.isnull().sum())
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            fig = px.imshow(numeric_df.corr(), text_auto=".2f")
            st.plotly_chart(fig, use_container_width=True)
        
        # Interactive charts for distributions
        st.subheader("Distribution Charts")
        col = st.selectbox("Pick column", df.columns)
        st.plotly_chart(px.histogram(df, x=col), use_container_width=True)

    with tabs[1]: # Training
        target_col = st.selectbox("🎯 Target Column", df.columns)
        
        if task_type == "Auto-Detect":
            task_type = "Classification" if df[target_col].nunique() < 20 else "Regression"
            st.info(f"Auto-Detected: {task_type}")
        
        if st.button("🚀 Train Models", type="primary"):
            log_area = st.empty()
            start_time = time.time()
            log_area.info("Training started...")
            
            # Data prep
            df_clean = df.dropna(subset=[target_col])
            X = df_clean.drop(columns=[target_col])
            y = df_clean[target_col]
            for col in X.select_dtypes(include='object'): X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            X = SimpleImputer().fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=n_estimators) if task_type=="Classification" else RandomForestRegressor(n_estimators=n_estimators),
                'Linear/Logistic': LogisticRegression(max_iter=1000) if task_type=="Classification" else LinearRegression()
            }
            
            results = {}
            for name, model in models.items():
                log_area.info(f"Training {name}...")
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                score = accuracy_score(y_test, pred) if task_type=="Classification" else r2_score(y_test, pred)
                results[name] = {"model": model, "score": score, "cv": cv_scores.mean()}
            
            best_name = max(results, key=lambda x: results[x]['score'])
            st.session_state['results'] = results
            st.session_state['best_model'] = results[best_name]['model']
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            
            # Save to experiment history
            st.session_state['experiment_history'].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "model": best_name,
                "score": results[best_name]['score']
            })
            notif.success(f"Training complete! Best: {best_name}")
            log_area.success(f"Done in {time.time()-start_time:.2f}s")

    with tabs[2]: # Leaderboard
        if 'results' in st.session_state:
            st.subheader("Model Leaderboard")
            lb = pd.DataFrame([{ "Model": k, "Score": v['score'], "CV Mean": v['cv']} for k,v in st.session_state['results'].items()])
            st.dataframe(lb.sort_values("Score", ascending=False), use_container_width=True)

    with tabs[3]: # Explainability
        if 'best_model' in st.session_state:
            st.subheader("SHAP Feature Importance Chart")
            explainer = shap.Explainer(st.session_state['best_model'], st.session_state['X_test'][:50])
            shap_values = explainer(st.session_state['X_test'][:50])
            fig = px.bar(x=df.columns[:-1], y=np.abs(shap_values.values).mean(0))
            st.plotly_chart(fig)

    with tabs[4]: # Project management + history
        st.subheader("Project Management")
        project_name = st.text_input("Project Name")
        if st.button("💾 Save Project"):
            with open(f"{PROJECT_DIR}/{project_name}.json", 'w') as f:
                json.dump(st.session_state['experiment_history'], f)
            notif.success("Project Saved!")
        
        st.subheader("Recent Projects Dashboard")
        files = os.listdir(PROJECT_DIR)
        st.selectbox("Open Project", files if files else ["No projects"])
        
        st.subheader("Experiment History")
        st.dataframe(pd.DataFrame(st.session_state['experiment_history']))

    with tabs[5]: # Export
        if 'best_model' in st.session_state:
            buf = io.BytesIO()
            pickle.dump(st.session_state['best_model'], buf)
            st.download_button("📦 Download Model", buf.getvalue(), "model.pkl")
            
            st.subheader("One-click deployment")
            st.code("docker run -p 8501:8501 your-app\n# or deploy to Streamlit Cloud")

else:
    st.info("Upload CSV to start. Press `Ctrl+R` for Keyboard Shortcut to refresh")
