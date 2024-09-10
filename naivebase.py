import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class MultinomialNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_feature_count = {c: np.zeros(X.shape[1]) for c in self.classes}
        self.class_count = {c: 0 for c in self.classes}
        self.prior = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.class_feature_count[c] = X_c.sum(axis=0)
            self.class_count[c] = X_c.shape[0]
        
        total_count = len(y)
        for c in self.classes:
            self.prior[c] = self.class_count[c] / total_count

    def calculate_likelihood(self, feature_value, class_value):
        likelihood = (self.class_feature_count[class_value] + 1) / (np.sum(self.class_feature_count[class_value]) + len(self.class_feature_count[class_value]))
        return likelihood ** feature_value

    def calculate_posterior(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.prior[c])
            likelihood = np.sum(np.log(self.calculate_likelihood(x, c)))
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self.calculate_posterior(x) for x in X])

# Streamlit app
st.title("Multinomial Naive Bayes Classifier")

# Load the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.write("Dataset:")
    st.write(data.head())

    # Select target variable
    target_column = st.selectbox("Select target column", data.columns)
    features = [col for col in data.columns if col != target_column]

    # Feature selection
    selected_features = st.multiselect("Select features", features, default=features)
    
    if len(selected_features) == 0:
        st.error("Please select at least one feature.")
    else:
        X = data[selected_features].values
        y = data[target_column].values
        
        # Encode target variable if it's categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            class_names = le.classes_
        else:
            class_names = np.unique(y)

        # Train-test split
        split_ratio = st.slider("Train-test split ratio", 0.5, 0.9, 0.7)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=42)

        # Train the model
        model = MultinomialNaiveBayes()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        st.write(f"Accuracy: {accuracy * 100:.2f}%")

        # Show predictions
        if st.checkbox("Show Predictions"):
            st.write("Predictions:", [class_names[int(pred)] for pred in y_pred])
            st.write("True values:", [class_names[int(true)] for true in y_test])
