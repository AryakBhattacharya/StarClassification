import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import time
import io
import joblib
import gradio as gr
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from collections import Counter

filename = "Star99999_raw.csv"
df = pd.read_csv(filename)
print("Loaded:", filename)
df.head()

# Dropping Index Column
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Replacing placeholders and dropping
placeholders = ["", "/", "...", "....", ".....", "--", "—"]
df.replace(placeholders, np.nan, inplace=True)
df.dropna(inplace=True)

# Converting to float value
numeric_cols = ['Vmag', 'Plx', 'e_Plx', 'B-V']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Dropping any rows that failed to convert to float
df.dropna(subset=numeric_cols, inplace=True)

# Filtering Parallax Error
# Keeping rows with relative error e_Plx/Plx <= 0.2
df = df[df['Plx'] > 0]
df = df[df['e_Plx'] / df['Plx'] <= 0.2]

# Cleaning the SpType column and dropping not suitable columns
def clean_sptype(s):
    if pd.isna(s): return np.nan
    s = s.strip().replace(':', '').replace('?', '').replace('...', '')
    match = re.match(r'^([OBAFGKM][0-9]?[IV]*)', s)
    return match.group(1) if match else np.nan

df['SpType'] = df['SpType'].apply(clean_sptype)
df.dropna(subset=['SpType'], inplace=True)

# Reseting Index
df.reset_index(drop=True, inplace=True)
print("Cleaned dataset shape:", df.shape)

# Saving the preprocessed data
df.to_csv("Cleaned data.csv", index=False)
print("Saved cleaned dataset as 'Cleaned data.csv'")

# Calculating distance in parsecs
df['Distance_pc'] = 1000 / df['Plx']

# Defining Spectral and Luminosity values
spectral_mapping = {
    'O': 0, 'B': 1, 'A': 2, 'F': 3, 'G': 4, 'K': 5, 'M': 6
}

luminosity_mapping = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'Unknown': 6
}

# Extracting Spectral and Luminosity
def extract_classes(sptype):
    sptype = str(sptype).upper()
    spectral = re.findall(r'[OBAFGKM]', sptype)
    luminosity = re.findall(r'\bI{1,3}|IV|V\b', sptype)

    spectral_class = spectral[0] if spectral else 'Unknown'
    luminosity_class = luminosity[0] if luminosity else 'Unknown'

    spectral_label = spectral_mapping.get(spectral_class, -1)
    luminosity_label = luminosity_mapping.get(luminosity_class, 6)

    return pd.Series([spectral_class, spectral_label, luminosity_class, luminosity_label])

df[['Spectral_Class', 'Spectral_Label', 'Luminosity_Class', 'Luminosity_Label']] = df['SpType'].apply(extract_classes)

# Dropping unknowns
#df = df[~((df['Spectral_Class'] == 'Unknown') & (df['Luminosity_Class'] == 'Unknown'))].copy()
#display(df)

# Preparing features and targets
# We can change 'y' to 'Luminosity_Label' to classify luminosity instead
X = df[['Vmag', 'Distance_pc', 'B-V']]
y = df['Spectral_Label']

# Test-Train Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Normalizing and applying PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Evaluating model performance
def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    # Start CV timer
    start = time.time()
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    end = time.time()
    cv_runtime = end - start

    # Fitting the model and predicting
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Binary format for AUC (One-vs-Rest)
    classes = np.unique(y_train)
    y_test_bin = label_binarize(y_test, classes=classes)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    # Evaluating the performance
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr') if y_score is not None else 'N/A'

    # Output
    print(f"--- {name} ---")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Precision         : {prec:.4f}")
    print(f"Recall            : {rec:.4f}")
    print(f"F1-Score          : {f1:.4f}")
    print(f"AUC (OvR)         : {auc:.4f}")
    print(f"CV Mean Accuracy  : {cv_scores.mean():.4f}")
    print(f"CV Runtime (secs) : {cv_runtime:.2f}")
    print()

# Dropping classes with < 6 samples

min_samples = 6
class_counts = Counter(y_train)
valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples]

# Filtering again
mask = y_train.isin(valid_classes)
X_train_filtered = X_train_pca[mask]
y_train_filtered = y_train[mask]

#Using SMOTE to balance distribution

# Checking class distribution first
print("Before SMOTE:\n", y_train.value_counts())

# Applying SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_filtered, y_train_filtered)

# Confirming the new balanced distribution
print("\nAfter SMOTE:\n", pd.Series(y_train_balanced).value_counts())

# Training Models
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='rbf', probability=True, random_state=42)

evaluate_model(dt_model, X_train_pca, y_train, X_test_pca, y_test, name="Decision Tree")
evaluate_model(rf_model, X_train_pca, y_train, X_test_pca, y_test, name="Random Forest")
evaluate_model(svm_model, X_train_pca, y_train, X_test_pca, y_test, name="SVM (RBF)")

# Visualizing PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='tab10', s=20, alpha=0.7)
plt.title("PCA - Training Set Visualization (Colored by Spectral Label)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar()
plt.show()

# Training the final model on full balanced training data
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train_balanced, y_train_balanced)

# Saving the model
joblib.dump(final_model, "star_classifier_model.pkl")
print("✅ Model saved to star_classifier_model.pkl")

# Saving scaler and PCA for reuse in GUI
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

# Re-mapping from label to spectral class
label_to_spectral = {
    0: 'O', 1: 'B', 2: 'A', 3: 'F', 4: 'G', 5: 'K', 6: 'M'
}

# Adding descriptions for each spectral class
spectral_descriptions = {
    'O': "Very hot, blue star",
    'B': "Hot, blue-white star",
    'A': "White star",
    'F': "Yellow-white star",
    'G': "Sun-like star",
    'K': "Orange star",
    'M': "Cool, red star"
}

# Loading the model and preprocessors
model = joblib.load("star_classifier_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# Checking an example prediction
sample = pd.DataFrame([[5.0, 50.0, 0.7]], columns=['Vmag', 'Distance_pc', 'B-V'])
sample_scaled = scaler.transform(sample)
sample_pca = pca.transform(sample_scaled)
prediction = model.predict(sample_pca)

# Mapping to class and description
pred_class = label_to_spectral.get(prediction[0], "Unknown")
pred_desc = spectral_descriptions.get(pred_class, "Unknown spectral type")

print(f"🔮 Predicted Spectral Class: {pred_class} → {pred_desc}")

# Loading the model and preprocessing components
model = joblib.load("star_classifier_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# Gradio prediction function
def predict_star_class(vmag, bv, plx):
    try:
        distance = 1000 / plx
        features = pd.DataFrame([[vmag, distance, bv]], columns=['Vmag', 'Distance_pc', 'B-V'])
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        pred = model.predict(features_pca)

        pred_class = label_to_spectral.get(pred[0], "Unknown")
        pred_desc = spectral_descriptions.get(pred_class, "Unknown spectral type")
        return f"🔭 Predicted Spectral Class: {pred_class} → {pred_desc}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Launch Gradio app
gr.Interface(
    fn=predict_star_class,
    inputs=[
        gr.Number(label="Vmag"),
        gr.Number(label="B-V"),
        gr.Number(label="Plx (mas)")
    ],
    outputs="text",
    title="Star Spectral Class Predictor",
    description="Enter star features to predict its spectral class and description."
).launch()