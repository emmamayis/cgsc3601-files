import pandas as pd
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: Load the preprocessed dataset ===
df = pd.read_csv("finaldataset2.csv").astype(str)

# === Step 2: Force specific rare samples into the training set ===
# Select 1 participant for each important group
high_anxiety_yes = df[
    (df['anxietymode'].str.lower() == 'high') &
    (df['risk_averse'].str.lower() == 'yes')
].sample(1, random_state=42)

high_anxiety_no = df[
    (df['anxietymode'].str.lower() == 'high') &
    (df['risk_averse'].str.lower() == 'no')
].sample(1, random_state=42)

low_anxiety_yes = df[
    (df['anxietymode'].str.lower() == 'low') &
    (df['risk_averse'].str.lower() == 'yes')
].sample(1, random_state=42)

low_anxiety_no = df[
    (df['anxietymode'].str.lower() == 'low') &
    (df['risk_averse'].str.lower() == 'no')
].sample(1, random_state=42)

average_anxiety_yes = df[
    (df['anxietymode'].str.lower() == 'average') &
    (df['risk_averse'].str.lower() == 'yes')
].sample(1, random_state=42)

average_anxiety_no = df[
    (df['anxietymode'].str.lower() == 'average') &
    (df['risk_averse'].str.lower() == 'no')
].sample(1, random_state=42)

# Combine all six forced samples into one DataFrame
forced_samples = pd.concat([
    high_anxiety_yes,
    high_anxiety_no,
    low_anxiety_yes,
    low_anxiety_no,
    average_anxiety_yes,
    average_anxiety_no
])

# === Step 3: Remove forced samples from the full dataset ===
remaining = df.drop(forced_samples.index)

# === Step 4: Randomly split the remaining data into 20 test samples ===
remaining_train, test_set = train_test_split(
    remaining,
    test_size=20,  # manually force exactly 20 test samples
    random_state=42,
    shuffle=True
)

# === Step 5: Finalize the training set ===
train_set = pd.concat([remaining_train, forced_samples]).reset_index(drop=True)

print(f"\n=== Training Set Size: {len(train_set)} samples ===")
print(f"=== Testing Set Size: {len(test_set)} samples ===")

# === Step 6: Define the Bayesian Network structure ===
model = BayesianModel([
    ("culture", "risk_averse"),
    ("anxietymode", "risk_averse")
])

# Train the Bayesian Network
model.fit(train_set, estimator=MaximumLikelihoodEstimator)

# === Step 7: Print the learned Conditional Probability Tables (CPTs) ===
print("\n=== Learned CPTs from Training Data ===")
for cpd in model.get_cpds():
    print(cpd)
    print("\n" + "="*50 + "\n")

# === Step 8: Set up the inference engine ===
inference = VariableElimination(model)

# === Step 9: Test the model on unseen test samples ===
correct = 0
wrong = 0
total = 0

# For confusion matrix and probability analysis
y_true = []
y_pred = []
y_prob = []

print("\n=== Testing on Test Samples ===")

for idx, row in test_set.iterrows():
    print(f"\n--- Test Sample {idx} ---")
    print(f"Input Evidence: culture = {row['culture']}, anxietymode = {row['anxietymode']}")

    # Predict risk_averse
    result = inference.query(
        variables=["risk_averse"],
        evidence={
            "culture": row["culture"],
            "anxietymode": row["anxietymode"]
        }
    )
    print(result)

    predicted_index = result.values.argmax()
    predicted_label = "yes" if predicted_index == 1 else "no"
    actual_label = row["risk_averse"].lower()

    # Save results
    y_true.append(actual_label)
    y_pred.append(predicted_label)
    y_prob.append(result.values[predicted_index])

    print(f"Predicted: {predicted_label} (Confidence: {result.values[predicted_index]:.2f})")
    print(f"Actual: {actual_label}")

    if predicted_label == actual_label:
        print("✅ Correct")
        correct += 1
    else:
        print("❌ Incorrect")
        wrong += 1

    total += 1

# === Step 10: Calculate final accuracy ===
accuracy = (correct / total) * 100

print("\n=== Final Accuracy Report ===")
print(f"Total Test Samples: {total}")
print(f"Correct Predictions: {correct}")
print(f"Wrong Predictions: {wrong}")
print(f"Accuracy: {accuracy:.2f}%")

# === Step 11: Confusion Matrix and Classification Report ===
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_true, y_pred, labels=["yes", "no"])
print(cm)

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=["yes", "no"]))

# === Step 12: Plot Confusion Matrix Heatmap ===
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["yes", "no"], yticklabels=["yes", "no"])
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# === Step 13: Plot Test Accuracy Bar Chart ===
plt.bar(["Test Accuracy"], [accuracy], color='skyblue')
plt.ylim(0, 100)
plt.title(f"Model Test Accuracy: {accuracy:.2f}%")
plt.ylabel("Accuracy (%)")
plt.show()
