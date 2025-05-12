import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Customer Campaign Prediction", layout="wide")
st.title("Customer Campaign Response Prediction")

# Load data
def load_data():
    df = pd.read_excel("marketing_campaign.xlsx", sheet_name="Sheet1")
    columns_to_drop = [col for col in ['ID', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue'] if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    df = df.dropna()
    return df

def preprocess_data(df):
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop(columns=['Response'])
    y = df['Response']
    return X, y, df

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    # Feature importance: absolute value of coefficients
    import numpy as np
    importance = np.abs(model.coef_[0])
    return model, scaler, accuracy, X_train.columns, importance

def main():
    df = load_data()
    X, y, processed_df = preprocess_data(df)
    model, scaler, accuracy, feature_names, importance = train_model(X, y)

    # Select top N most important features
    import numpy as np
    N = 5
    coefs = model.coef_[0]
    top_idx = importance.argsort()[::-1][:N]
    top_features = [feature_names[i] for i in top_idx]
    st.header("Top Factors Affecting Campaign Response Prediction")
    for i, idx in enumerate(top_idx):
        feat = feature_names[idx]
        coef = coefs[idx]
        direction = "Higher values increase likelihood to respond" if coef > 0 else "Higher values decrease likelihood to respond"
        min_val = float(df[feat].min()) if feat in df.columns else 0.0
        max_val = float(df[feat].max()) if feat in df.columns else 1.0
        st.markdown(f"**{i+1}. {feat}**")
        st.write(f"- Direction: {direction}")
        st.write(f"- Allowed range: {min_val} to {max_val}")
    st.markdown(":bulb: Only these top features are used for prediction input below. Their importance and direction are based on the model's coefficients.")

    st.header("Input Customer Data for Prediction (Top Factors Only)")
    input_data = {}
    for col in top_features:
        if "_" in col and col.split("_")[0] in df.columns and df[col.split("_")[0]].dtype == object:
            base_col = col.split("_")[0]
            options = [c.replace(f"{base_col}_", "") for c in X.columns if c.startswith(f"{base_col}_")]
            input_data[col] = st.selectbox(f"{base_col}", options)
        else:
            min_val = float(df[col].min()) if col in df.columns else 0.0
            max_val = float(df[col].max()) if col in df.columns else 1.0
            input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=min_val)

    if st.button("Predict Response"):
        # Build input vector for all features, fill with 0/default except for top features
        input_vector = []
        for col in X.columns:
            if col in input_data:
                if isinstance(input_data[col], str):
                    input_vector.append(1 if input_data[col] == col.split("_")[1] else 0)
                else:
                    input_vector.append(input_data.get(col, 0))
            else:
                input_vector.append(0)
        input_vector = pd.DataFrame([input_vector], columns=X.columns)
        input_scaled = scaler.transform(input_vector)
        prediction = model.predict(input_scaled)[0]

        # Reasoning for prediction
        reasons = []
        for idx in top_idx:
            feat = feature_names[idx]
            coef = coefs[idx]
            user_val = input_vector.iloc[0][feat]
            min_val = float(df[feat].min()) if feat in df.columns else 0.0
            max_val = float(df[feat].max()) if feat in df.columns else 1.0
            if coef > 0:
                if user_val > (max_val + min_val)/2:
                    reasons.append(f"{feat} is high (which increases chance of response)")
                else:
                    reasons.append(f"{feat} is low (which decreases chance of response)")
            else:
                if user_val > (max_val + min_val)/2:
                    reasons.append(f"{feat} is high (which decreases chance of response)")
                else:
                    reasons.append(f"{feat} is low (which increases chance of response)")
        st.subheader("Prediction Result:")
        if prediction == 1:
            st.success("Customer will respond to the campaign.")
            st.info("Reason: " + "; ".join([r for r in reasons if "increases" in r]))
        else:
            st.error("Customer will NOT respond to the campaign.")
            st.info("Reason: " + "; ".join([r for r in reasons if "decreases" in r]))


if __name__ == "__main__":
    main()
