import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io

st.set_page_config(page_title="Student Performance Dashboard", layout="wide", page_icon="🎓")

# -- 1. Data Generation or Loading --
def create_dummy_data():
    """Generates dummy dataset if no CSV is found/uploaded."""
    np.random.seed(42)
    n = 500
    data = {
        'AcademicScore': np.random.normal(70, 15, n).clip(0, 100),
        'CourseParticipation': np.random.normal(50, 20, n).clip(0, 100),
        'AttendanceRate': np.random.normal(85, 10, n).clip(0, 100),
        'PhysicalActivity': np.random.normal(3, 1.5, n).clip(0, 7),
        'EmotionEngagement': np.random.choice(['Low', 'Medium', 'High'], n),
        'LearningStyle': np.random.choice(['Visual', 'Auditory', 'Kinesthetic'], n),
        'DeviceUsage': np.random.normal(4, 2, n).clip(0, 12),
        'FeedbackScore': np.random.normal(7, 2, n).clip(0, 10)
    }
    df = pd.DataFrame(data)
    
    # Calculate a score to determine performance
    score = (df['AcademicScore'] * 0.4 + 
             df['AttendanceRate'] * 0.3 + 
             df['CourseParticipation'] * 0.2 + 
             df['FeedbackScore'] * 10 * 0.1)
    
    conditions = [
        (score >= 80),
        (score >= 60) & (score < 80),
        (score < 60)
    ]
    choices = ['High', 'Mid', 'Low']
    df['StudentPerformance'] = np.select(conditions, choices, default='Mid')
    return df

@st.cache_data
def load_data(uploaded_file=None):
    """Loads CSV data or creates dummy data."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("student_education_dataset.csv")
            return df
        except FileNotFoundError:
            # Fallback to dummy data
            df = create_dummy_data()
            df.to_csv("student_education_dataset.csv", index=False)
            return df

# -- 2. Preprocessing Data --
def preprocess_data(df):
    """Preprocess data for ML modeling."""
    df_processed = df.copy()
    
    categorical_cols = ['EmotionEngagement', 'LearningStyle']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
            
    target_le = LabelEncoder()
    # Ensure classes are mapped conceptually if applicable
    if 'StudentPerformance' in df_processed.columns:
        df_processed['StudentPerformance'] = target_le.fit_transform(df_processed['StudentPerformance'].astype(str))
        
    return df_processed, label_encoders, target_le

@st.cache_resource
def train_models(df_processed):
    """Train ML Models."""
    X = df_processed.drop('StudentPerformance', axis=1)
    y = df_processed['StudentPerformance']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return lr_model, rf_model, scaler, X_train, X_test, y_train, y_test, X, y

# -- 3. Build UI Layout --
st.title("🎓 Student Performance Data Mining Dashboard")
st.markdown("A modern interactive dashboard replicating a full machine learning pipeline.")
st.markdown("---")

# Sidebar
st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio("Go to:", [
    "1. Overview", 
    "2. Exploratory Data Analysis (EDA)", 
    "3. Model Performance", 
    "4. Prediction"
])

st.sidebar.markdown("---")
st.sidebar.header("📁 Data Upload")
st.sidebar.write("Upload new data to automatically retrain the models.")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    st.sidebar.success("File successfully uploaded! Retraining models...")

# Load Data
df = load_data(uploaded_file)

if df.empty:
    st.error("Dataset is empty. Please upload a valid CSV.")
else:
    # Preprocessing
    df_processed, label_encoders, target_le = preprocess_data(df)
    
    # Retrain model seamlessly based on data
    lr_model, rf_model, scaler, X_train, X_test, y_train, y_test, X, y = train_models(df_processed)
    
    # --- OVERVIEW SECTION ---
    if menu == "1. Overview":
        st.header("📊 Dataset Overview")
        st.write("This section provides a quick glance at the underlying dataset structure.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Rows:** {df.shape[0]} \n\n **Columns:** {df.shape[1]}")
            
        with col2:
            st.write("**Class Distribution (Low, Mid, High)**")
            if 'StudentPerformance' in df.columns:
                class_counts = df['StudentPerformance'].value_counts()
                st.bar_chart(class_counts, color="#636EFA")
            else:
                st.warning("Column 'StudentPerformance' not found.")
            
        st.subheader("Dataset Peek")
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    # --- EDA SECTION ---
    elif menu == "2. Exploratory Data Analysis (EDA)":
        st.header("📈 Exploratory Data Analysis (EDA)")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Correlations", "Distributions", "Boxplots", "Learning Styles"])
        
        with tab1:
            st.subheader("Correlation Heatmap")
            st.write("Understand linear relationships between input features.")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_processed.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
            st.pyplot(fig)
            
        with tab2:
            st.subheader("Feature Distributions")
            st.write("Examine the spread and shape of continuous data.")
            numeric_cols = df.select_dtypes(include=np.number).columns
            feature_to_plot = st.selectbox("Select a feature", numeric_cols)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[feature_to_plot], kde=True, ax=ax, color='teal')
            st.pyplot(fig)
            
        with tab3:
            st.subheader("Boxplots for Key Features")
            st.write("Analyze distributions grouped by performance to detect patterns.")
            box_feature = st.selectbox("Select a feature for boxplot", numeric_cols, index=0)
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
                st.write("LearningStyle column is not present in the dataset.")
            
    # --- MODEL PERFORMANCE SECTION ---
    elif menu == "3. Model Performance":
        st.header("⚙️ Model Performance")
        
        model_choice = st.radio("Select Model to Evaluate", ["Logistic Regression", "Random Forest"], horizontal=True)
        
        if model_choice == "Logistic Regression":
            model = lr_model
        else:
            model = rf_model
            
        y_pred = model.predict(X_test)
        
        st.markdown("### Metrics Overview")
        col1, col2 = st.columns(2)
        with col1:
            acc = accuracy_score(y_test, y_pred)
            st.metric(label="Testing Accuracy", value=f"{acc*100:.2f}%")
            
        with col2:
            cv_scores = cross_val_score(model, X_scaled_df, y, cv=5)
            st.metric(label="5-Fold CV Mean Accuracy", value=f"{cv_scores.mean()*100:.2f}%")
            
        st.markdown("---")
        
        col_report, col_cm = st.columns(2)
        with col_report:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=target_le.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='Blues'), use_container_width=True)
            
        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="PuBuGn", xticklabels=target_le.classes_, yticklabels=target_le.classes_, ax=ax)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
            
        if model_choice == "Random Forest":
            st.markdown("---")
            st.subheader("🌲 Feature Importance")
            importances = model.feature_importances_
            feature_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=feature_imp.values, y=feature_imp.index, ax=ax, palette="viridis")
            ax.set_title("Random Forest Feature Importance")
            ax.set_xlabel("Importance Score")
            st.pyplot(fig)
            
    # --- PREDICTION SECTION ---
    elif menu == "4. Prediction":
        st.header("🔮 Predict New Student Performance")
        st.write("Enter student details below to predict their performance class dynamically. Adjust values to see how different parameters weigh in.")
        
        with st.form("prediction_form", clear_on_submit=False):
            st.subheader("Input Features")
            col1, col2 = st.columns(2)
            
            inputs = {}
            for i, col in enumerate(X.columns):
                target_col = col1 if i % 2 == 0 else col2
                with target_col:
                    if col in label_encoders:
                        classes = label_encoders[col].classes_
                        inputs[col] = st.selectbox(f"{col}", classes)
                    else:
                        min_val = float(df[col].min())
                        max_val = float(df[col].max())
                        mean_val = float(df[col].mean())
                        inputs[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
                        
            submit_button = st.form_submit_button(label="Predict Performance", use_container_width=True)
            
        if submit_button:
            input_df = pd.DataFrame([inputs])
            
            for col, le in label_encoders.items():
                if col in input_df.columns:
                    input_df[col] = le.transform(input_df[col].astype(str))
                    
            input_scaled = scaler.transform(input_df)
            
            # Prediction via Random Forest
            prediction = rf_model.predict(input_scaled)
            pred_class = target_le.inverse_transform(prediction)[0]
            
            probabilities = rf_model.predict_proba(input_scaled)[0]
            confidence = max(probabilities) * 100
            
            st.markdown("---")
            st.subheader("Assessment Result")
            st.success(f"### Predicted Class: **{pred_class}**")
            st.info(f"### Confidence Score: **{confidence:.1f}%**")
            
            # Display probabilities
            st.write("**Prediction Probabilities across classes**")
            prob_df = pd.DataFrame({
                "Performance Class": target_le.classes_,
                "Probability": probabilities
            })
            st.bar_chart(prob_df.set_index("Performance Class"), color="#00C4A7")
