import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import io, pickle, json, shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from fpdf import FPDF

st.set_page_config(page_title="AI Data Scientist Copilot Pro", layout="wide")

# === SESSION STATE MANAGEMENT + LOGGING ===
if 'df' not in st.session_state: st.session_state.df = None
if 'model' not in st.session_state: st.session_state.model = None
if 'logs' not in st.session_state: st.session_state.logs = []

def log(msg): st.session_state.logs.append(msg)

# === DARK MODE + RESPONSIVE UI ===
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)
if dark_mode: st.markdown("""<style>.stApp {background: #0e1117; color: white;}</style>""", unsafe_allow_html=True)

# === API KEY CONFIGURATION ===
with st.sidebar:
    st.header("⚙️ Settings")
    openai_key = st.text_input("OpenAI API Key", type="password")

st.title("🤖 AI Data Scientist Copilot Pro")
st.caption("End-to-End Workflow Integration: Upload → Clean → Analyze → Model → Explain → Export")

tab1, tab2, tab3, tab4 = st.tabs(["1. Upload & Clean", "2. EDA & Charts", "3. AutoML Engine", "4. AI + Export"])

# === TAB 1: DATASET UPLOAD BACKEND + DATA CLEANING PIPELINE ===
with tab1:
    st.header("Dataset Upload Backend + Validation")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file:
        with st.spinner("Validating and Processing..."):
            log("File Uploaded")
            df = pd.read_csv(uploaded_file)
            
            if df.empty: st.error("File is empty"); st.stop()
            
            st.session_state.df = df
            log("Data Validation Passed")
        
        # KPI CARDS
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows", df.shape[0]); c2.metric("Cols", df.shape[1])
        c3.metric("Missing", df.isnull().sum().sum()); c4.metric("Duplicates", df.duplicated().sum())
        
        # DATASET PREVIEW
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # DATA CLEANING PIPELINE
        if st.button("Run Full Cleaning Pipeline"):
            with st.spinner("Running Pipeline..."):
                report = {}
                df = df.fillna(df.mean(numeric_only=True))
                df = df.drop_duplicates()
                
                num_df = df.select_dtypes(include=np.number)
                if not num_df.empty:
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    df['Outlier'] = iso.fit_predict(num_df)
                    report['Outliers Found'] = (df['Outlier']==-1).sum()
                
                for col in num_df.columns:
                    df[f'{col}_log'] = np.log1p(df[col])
                
                st.session_state.df = df
                report['Final Shape'] = df.shape
                st.success("Cleaning Complete!")
                st.json(report)

# === TAB 2: EDA + INTERACTIVE CHARTS ENGINE + PLOTLY ===
with tab2:
    st.header("Exploratory Data Analysis")
    if st.session_state.df is not None:
        df = st.session_state.df
        num_df = df.select_dtypes(include=np.number)
        
        st.subheader("Correlation Matrix")
        if num_df.shape[1] > 1:
            corr = num_df.corr()
            fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need 2+ numeric columns for correlation")
        
        st.subheader("Interactive Charts Engine")
        chart_type = st.selectbox("Chart Type", ["Histogram", "Scatter", "Box"])
        col = st.selectbox("Column", df.columns)
        if chart_type == "Histogram": fig = px.histogram(df, x=col)
        if chart_type == "Scatter" and len(num_df.columns)>1: fig = px.scatter(df, x=num_df.columns[0], y=num_df.columns[1])
        if chart_type == "Box": fig = px.box(df, y=col)
        st.plotly_chart(fig)
    else: st.info("Upload data first")

# === TAB 3: AUTOML ENGINE - FIXED FOR DATE/TEXT + FEATURE IMPORTANCE ===
with tab3:
    st.header("AutoML Engine")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        target = st.selectbox("Target Column", df.columns)
        problem_type = st.radio("Problem Type", ["Auto Detect", "Classification", "Regression"])
        
        if st.button("Train & Compare Models"):
            with st.spinner("Training Pipeline Running..."):
                # === CLEAN DATA FOR ML ===
                X = df.drop(columns=[target])
                y = df[target]
                
                # Drop Date columns and non-numeric columns from X
                X = X.select_dtypes(include=[np.number]) 
                
                # Handle text/date target
                if y.dtype == 'object':
                    try:
                        y = pd.to_datetime(y)
                        y = y.astype('int64') // 10**9
                    except:
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                
                # Drop rows with NaN
                data = pd.concat([X, y], axis=1).dropna()
                X = data[X.columns]
                y = data[y.name]
                
                if X.shape[1] == 0:
                    st.error("No numeric features left after dropping Date/Text columns. Please select a numeric target like Sales, Price, etc.")
                    st.stop()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                if problem_type == "Auto Detect":
                    problem_type = "Classification" if y.nunique() < 20 else "Regression"
                
                results = {}
                if problem_type == "Regression":
                    models = {"Linear": LinearRegression(), "RF": RandomForestRegressor(random_state=42)}
                    metric = r2_score
                else:
                    models = {"Logistic": LogisticRegression(max_iter=500), "RF": RandomForestClassifier(random_state=42)}
                    metric = accuracy_score
                
                for name, model in models.items():
                    params = {'n_estimators':[50,100]} if "RF" in name else {}
                    grid = GridSearchCV(model, params, cv=3).fit(X_train, y_train)
                    pred = grid.predict(X_test)
                    results[name] = metric(y_test, pred)
                    if results[name] == max(results.values()): st.session_state.model = grid.best_estimator_
                
                st.success(f"Best Model: {max(results, key=results.get)} Score: {max(results.values()):.3f}")
                st.bar_chart(pd.DataFrame.from_dict(results, orient='index'))
                
                if problem_type == "Classification" and len(np.unique(y)) == 2:
                    cm = confusion_matrix(y_test, pred)
                    st.subheader("Confusion Matrix")
                    st.write(cm)
                    try:
                        fpr, tpr, _ = roc_curve(y_test, grid.predict_proba(X_test)[:,1])
                        st.plotly_chart(px.area(x=fpr, y=tpr, title=f"ROC AUC={auc(fpr,tpr):.2f}"))
                    except: pass
                
                # === FIXED FEATURE IMPORTANCE SECTION ===
                st.subheader("Top 10 Feature Importance")

                if hasattr(st.session_state.model, 'feature_importances_'):
                    importances = pd.Series(st.session_state.model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
                    st.bar_chart(importances)
                    
                elif hasattr(st.session_state.model, 'coef_'):
                    coef = st.session_state.model.coef_
                    if len(coef.shape) > 1: # for classification
                        coef = coef[0]
                    importances = pd.Series(np.abs(coef), index=X.columns).sort_values(ascending=False)[:10]
                    st.bar_chart(importances)
                    st.caption("Showing |coefficients| for Linear/Logistic model")
                else:
                    st.info("This model doesn't support feature importance")
                
                st.download_button("Download Model", pickle.dumps(st.session_state.model), "model.pkl")
    else: st.info("Upload data first")

# === TAB 4: AI CHAT BACKEND + LLM INTEGRATION + PDF REPORT ===
with tab4:
    st.header("AI Chat + Report Generator")
    if st.session_state.df is not None:
        df = st.session_state.df
        num_df = df.select_dtypes(include=np.number)
        
        if st.button("Generate AI Insights"):
            if num_df.shape[1] > 1:
                top_corr = num_df.corr().iloc[0,1]
                insight = f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. Top correlation: {top_corr:.2f}"
            else:
                insight = f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns. Avg of {num_df.columns[0]}: {num_df.mean().iloc[0]:.2f}"
            st.info(f"AI Insight: {insight}")
        
        query = st.text_input("Ask about your data: 'What is average sales?'")
        if query:
            if "average" in query.lower() and not num_df.empty:
                st.write(f"AI Answer: Average of {num_df.columns[0]} is {num_df.mean().iloc[0]:.2f}")
            else:
                st.write("AI Answer: Upload OpenAI key for advanced answers")
        
        if st.button("Generate PDF Report"):
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="AI Data Copilot Report", ln=True, align='C')
            pdf.cell(200, 10, txt=f"Rows: {df.shape[0]}", ln=True)
            pdf.output("report.pdf")
            with open("report.pdf", "rb") as f: st.download_button("Download PDF", f, "report.pdf")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df.to_excel(writer, index=False)
        st.download_button("📥 Export Excel", output.getvalue(), "data.xlsx")
        
    else: st.info("Upload data first")

# === SIDEBAR: ACTIVITY LOGS ===
with st.sidebar:
    st.header("📁 Activity Logs")
    for l in st.session_state.logs[-5:]: st.write(f"- {l}")