import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Student Performance Data Mining Dashboard", layout="wide", page_icon="🎓")

def create_dummy_data():
    """Generates dummy dataset mimicking the docs exactly to ensure 1000 records."""
    np.random.seed(42)
    n = 1000
    data = {
        'AcademicScore': np.random.uniform(50, 99, n),
        'CourseParticipation': np.random.uniform(0, 49, n).astype(int),
        'AttendanceRate': np.random.uniform(0.500, 0.999, n),
        'PhysicalActivity': np.random.uniform(6, 4997, n).astype(int),
        'EmotionEngagement': np.random.uniform(0.001, 0.998, n),
        'LearningStyle': np.random.choice(['Visual', 'Auditory', 'Kinesthetic'], n, p=[0.325, 0.324, 0.351]),
        'DeviceUsage': np.random.uniform(0, 29, n).astype(int),
        'FeedbackScore': np.random.uniform(1.003, 4.999, n)
    }
    df = pd.DataFrame(data)
    
    # Calculate performance level explicitly based on AcademicScore per the documentation
    conditions = [
        (df['AcademicScore'] >= 80),
        (df['AcademicScore'] >= 60) & (df['AcademicScore'] < 80),
        (df['AcademicScore'] < 60)
    ]
    choices = [2, 1, 0] # 2=High, 1=Mid, 0=Low
    df['StudentPerformance'] = np.select(conditions, choices, default=1)
    return df

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("student_education_dataset.csv")
            return df
        except FileNotFoundError:
            df = create_dummy_data()
            df.to_csv("student_education_dataset.csv", index=False)
            return df

def preprocess_data(df):
    df_processed = df.copy()
    label_encoders = {}
    
    for col in df_processed.columns:
        if not pd.api.types.is_numeric_dtype(df_processed[col]):
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
            
    return df_processed, label_encoders

@st.cache_resource
def train_models(df_processed):
    X = df_processed.drop('StudentPerformance', axis=1)
    y = df_processed['StudentPerformance'].astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Stratified Split per Docs
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    return lr_model, rf_model, scaler, X_train, X_test, y_train, y_test, X, y

st.title("🎓 Student Performance Data Mining Dashboard")
st.markdown("A modern interactive dashboard replicating a full machine learning classification pipeline based on the Student Education Dataset.")
st.markdown("---")

st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio("Go to:", ["1. Overview", "2. Exploratory Data Analysis (EDA)", "3. Model Performance", "4. Prediction"])

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload your exact CSV file", type=["csv"])
df = load_data(uploaded_file)

if df.empty:
    st.error("Dataset is empty. Please upload a valid CSV.")
else:
    df_processed, label_encoders = preprocess_data(df)
    lr_model, rf_model, scaler, X_train, X_test, y_train, y_test, X, y = train_models(df_processed)
    
    PERF_LABELS = {0: 'Level 0 (Low)', 1: 'Level 1 (Mid)', 2: 'Level 2 (High)'}
    
    # --- OVERVIEW ---
    if menu == "1. Overview":
        st.header("📊 Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Records:** {df.shape[0]} | **Features:** {df.shape[1]}")
        with col2:
            st.write("**Class Distribution: Low(0), Mid(1), High(2)**")
            class_counts = df['StudentPerformance'].map(PERF_LABELS).value_counts().sort_index()
            st.bar_chart(class_counts, color="#636EFA")
            
        st.subheader("Dataset Peek")
        st.dataframe(df.head(), use_container_width=True)
        st.subheader("Summary Statistics")
        st.dataframe(df.describe().round(3), use_container_width=True)

    # --- EDA ---
    elif menu == "2. Exploratory Data Analysis (EDA)":
        st.header("📈 Exploratory Data Analysis (EDA)")
        tab1, tab2, tab3, tab4 = st.tabs(["Correlations", "Distributions", "Boxplots", "Learning Styles"])
        
        with tab1:
            st.subheader("Pearson Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_processed.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, vmin=-1, vmax=1)
            st.pyplot(fig)
            
        with tab2:
            st.subheader("Feature Distributions by Performance Level")
            cols = df.select_dtypes(include=np.number).columns.drop('StudentPerformance', errors='ignore')
            feat_to_plot = st.selectbox("Select a feature", cols)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(data=df, x=feat_to_plot, hue='StudentPerformance', kde=True, ax=ax, palette='Set2')
            st.pyplot(fig)
            
        with tab3:
            st.subheader("Boxplots for Key Features")
            box_feature = st.selectbox("Select a feature for boxplot", cols, index=0)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x='StudentPerformance', y=box_feature, data=df, ax=ax, palette="Set2")
            st.pyplot(fig)
            
        with tab4:
            st.subheader("LearningStyle vs StudentPerformance")
            if 'LearningStyle' in df.columns:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.countplot(data=df, x='LearningStyle', hue='StudentPerformance', ax=ax, palette="pastel")
                st.pyplot(fig)
            else:
                st.write("LearningStyle column is not present.")

    # --- MODEL PERFORMANCE ---
    elif menu == "3. Model Performance":
        st.header("⚙️ Model Performance")
        model_choice = st.radio("Select Classifier to Evaluate", ["Logistic Regression", "Random Forest"], horizontal=True)
        model = lr_model if model_choice == "Logistic Regression" else rf_model
        
        y_pred = model.predict(X_test)
        
        col1, col2 = st.columns(2)
        with col1:
            acc = accuracy_score(y_test, y_pred)
            st.metric(label="Testing Accuracy", value=f"{acc*100:.2f}%")
        with col2:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # Apply scaling to the full dataset for Cross Validation
            X_scaled_cv = pd.DataFrame(scaler.transform(X), columns=X.columns)
            cv_scores = cross_val_score(model, X_scaled_cv, y, cv=cv)
            st.metric(label="5-Fold CV Mean Accuracy", value=f"{cv_scores.mean()*100:.2f}%")
            
        st.markdown("---")
        col_report, col_cm = st.columns(2)
        with col_report:
            st.subheader("Classification Report")
            t_names = [f"Level {k}" for k in sorted(y.unique())]
            report = classification_report(y_test, y_pred, target_names=t_names, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'), use_container_width=True)
            
        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="PuBuGn", xticklabels=t_names, yticklabels=t_names, ax=ax)
            plt.xlabel('Predicted Class')
            plt.ylabel('Actual Class')
            st.pyplot(fig)
            
        if model_choice == "Random Forest":
            st.markdown("---")
            st.subheader("🌲 Feature Importance")
            importances = model.feature_importances_
            fi = pd.Series(importances, index=X.columns).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=fi.values, y=fi.index, ax=ax, palette="viridis")
            ax.set_title("Random Forest Feature Importance")
            st.pyplot(fig)

    # --- PREDICTION ---
    elif menu == "4. Prediction":
        st.header("🔮 Predict New Student Performance")
        st.write("Dynamic prediction targeting Student Performance levels 0, 1, or 2.")
        
        with st.form("prediction_form", clear_on_submit=False):
            st.subheader("Input Features")
            col1, col2 = st.columns(2)
            inputs = {}
            for i, col in enumerate(X.columns):
                target_col = col1 if i % 2 == 0 else col2
                with target_col:
                    if col in label_encoders:
                        inputs[col] = st.selectbox(f"{col}", label_encoders[col].classes_)
                    else:
                        mi = float(df[col].min())
                        ma = float(df[col].max())
                        me = float(df[col].mean())
                        inputs[col] = st.number_input(f"{col}", min_value=mi, max_value=ma, value=me)
                        
            submit = st.form_submit_button(label="Predict Performance", use_container_width=True)
            
        if submit:
            idf = pd.DataFrame([inputs])
            for col, le in label_encoders.items():
                if col in idf.columns:
                    idf[col] = le.transform(idf[col].astype(str))
                    
            idf_scaled = scaler.transform(idf)
            
            prediction = rf_model.predict(idf_scaled)[0]
            probabilities = rf_model.predict_proba(idf_scaled)[0]
            confidence = max(probabilities) * 100
            
            st.markdown("---")
            st.success(f"### Predicted Performance Classification: **{PERF_LABELS.get(prediction, prediction)}**")
            st.info(f"### Confidence Score: **{confidence:.1f}%**")
            
            st.write("**Prediction Probabilities across classes**")
            prob_df = pd.DataFrame({"Class": [PERF_LABELS.get(k, k) for k in rf_model.classes_], "Probability": probabilities})
            st.bar_chart(prob_df.set_index("Class"), color="#00C4A7")
