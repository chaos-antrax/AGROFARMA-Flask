import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, ndcg_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import os
from datetime import datetime

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load the dataset
file_path = 'data/optimized_dataset.csv'
data = pd.read_csv(file_path)

# ------------------------------------------------ Data Preprocessing

# Extract year and ensure Month is properly formatted
if 'Year' not in data.columns and 'Date' in data.columns:
    data['Year'] = pd.to_datetime(data['Date']).dt.year
    data['Month'] = pd.to_datetime(data['Date']).dt.month
elif 'Month' in data.columns:
    # Ensure Month is in the right format
    data['Month'] = data['Month'].astype(int)

# Feature engineering
data['Price_to_Count_Ratio'] = data['Avg_Price'] / (data['Data_Count'] + 1)  # Adding 1 to avoid division by zero
data['Month_Sin'] = np.sin(2 * np.pi * data['Month']/12)  # Cyclical encoding for month
data['Month_Cos'] = np.cos(2 * np.pi * data['Month']/12)  # Cyclical encoding for month

# Define categorical and numerical features
categorical_features = ['Month']
numerical_features = ['Avg_Price', 'Price_Std', 'Data_Count', 'Price_to_Count_Ratio', 'Month_Sin', 'Month_Cos']

# Check for missing values and impute if necessary
imputer = SimpleImputer(strategy='median')
data[numerical_features] = imputer.fit_transform(data[numerical_features])

# Standardize numerical features
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Define features (X) and target (y)
X = data[categorical_features + numerical_features]
y = data['Vegetable']

# Sample splitting - Choose ONE of these methods based on your data structure:

# OPTION 1: Stratified split (if temporal order isn't critical)
print("Using stratified train-test split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# OPTION 2: Time-based split (if temporal patterns are important and you have Year in your data)
# Uncomment the following if you prefer this approach
# if 'Year' in data.columns:
#     print("Using time-based train-test split...")
#     max_year = data['Year'].max()
#     train_mask = data['Year'] < max_year  # Use all data except the last year for training
#     test_mask = data['Year'] == max_year  # Use the last year for testing
#     
#     X_train, X_test = X[train_mask], X[test_mask]
#     y_train, y_test = y[train_mask], y[test_mask]
#     print(f"Training years: {data.loc[train_mask, 'Year'].unique()}")
#     print(f"Testing year: {max_year}")

# ------------------------------------------------ Model parameters

# CatBoost parameters - optimized from previous experiments
cat_params = {
    'iterations': 500,            # More iterations for better convergence
    'learning_rate': 0.05,        # Balanced learning rate
    'depth': 8,                   # Tree depth
    'l2_leaf_reg': 3,             # L2 regularization to prevent overfitting
    'random_strength': 1,         # Randomness for robustness
    'bootstrap_type': 'Bayesian', # Bayesian bootstrap for stability
    'loss_function': 'RMSE',      # Regression loss function
    'eval_metric': 'RMSE',        # Evaluation metric
    'random_seed': 42,            # For reproducibility
    'verbose': 100                # Print progress every 100 iterations
}

# ------------------------------------------------ Model development

# Initialize models and predictions
models = {}
predictions = []
feature_importances = {}

vegetables = y.unique()
print(f"Training models for {len(vegetables)} vegetables: {', '.join(vegetables)}")

# Train a separate CatBoostRegressor for each vegetable
for veg in vegetables:
    print(f"\nTraining model for: {veg}")
    
    # Binary target for the current vegetable
    y_train_binary = (y_train == veg).astype(int)
    y_test_binary = (y_test == veg).astype(int)
    
    # Create the CatBoost Pool
    train_pool = Pool(X_train, y_train_binary, cat_features=categorical_features)
    test_pool = Pool(X_test, y_test_binary, cat_features=categorical_features)
    
    # Initialize and train the CatBoostRegressor with optimized parameters
    cb_model = CatBoostRegressor(**cat_params)
    
    # Train the model
    cb_model.fit(train_pool)
    
    # Save the model
    model_path = f'models/cb_model_{veg}.cbm'
    cb_model.save_model(model_path)
    models[veg] = model_path
    
    # Store feature importances
    feature_importances[veg] = list(zip(X.columns, cb_model.feature_importances_))
    
    # Predict suitability for the current vegetable
    preds = cb_model.predict(test_pool)
    predictions.append(preds)
    
    # Print individual vegetable performance
    mse = mean_squared_error(y_test_binary, preds)
    binary_preds = (preds > 0.5).astype(int)
    accuracy = np.mean(binary_preds == y_test_binary)
    print(f"MSE for {veg}: {mse:.4f}, Binary Accuracy: {accuracy:.2%}")

# Combine predictions into a DataFrame
predictions_df = pd.DataFrame(predictions).T
predictions_df.columns = vegetables

# Apply softmax to normalize predictions
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Apply softmax row-wise to get probabilities
normalized_predictions = np.apply_along_axis(softmax, 1, predictions_df.values)
normalized_predictions_df = pd.DataFrame(normalized_predictions, columns=vegetables)

# ------------------------------------------------ Testing & Evaluation

# Map vegetable labels to indices for evaluation
label_to_index = {veg: idx for idx, veg in enumerate(vegetables)}
y_test_indices = y_test.map(label_to_index)

# Predict top-1 vegetable for each test sample
pred_top1 = normalized_predictions_df.values.argmax(axis=1)

# Calculate Top-1 Accuracy
top1_accuracy = (pred_top1 == y_test_indices).mean()
print(f"\nEvaluation Results:")
print(f"Top-1 Accuracy: {top1_accuracy:.2%}")

# Calculate Top-3 Accuracy (useful for recommendation systems)
top3_predictions = np.argsort(normalized_predictions_df.values, axis=1)[:, -3:]
top3_correct = [y_test_indices.iloc[i] in top3_predictions[i] for i in range(len(y_test_indices))]
top3_accuracy = np.mean(top3_correct)
print(f"Top-3 Accuracy: {top3_accuracy:.2%}")

# Create one-hot encoded ground truth
true_relevance = np.zeros((y_test.size, len(vegetables)))
true_relevance[np.arange(y_test.size), y_test_indices] = 1

# Calculate NDCG (Normalized Discounted Cumulative Gain)
ndcg = ndcg_score(true_relevance, normalized_predictions_df.values)
print(f"NDCG (Normalized Discounted Cumulative Gain): {ndcg:.2f}")

# Calculate MAP (Mean Average Precision)
map_score = average_precision_score(true_relevance, normalized_predictions_df.values, average='samples')
print(f"MAP (Mean Average Precision): {map_score:.2%}")

# Calculate Mean Squared Error of suitability scores
mse = mean_squared_error(true_relevance.flatten(), normalized_predictions_df.values.flatten())
print(f"Mean Squared Error (MSE) of suitability scores: {mse:.2f}")

# ------------------------------------------------ Output Results

# Print feature importances for each vegetable
print("\nFeature Importances:")
for veg, importances in feature_importances.items():
    print(f"\n{veg}:")
    sorted_importances = sorted(importances, key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importances:
        print(f"  {feature}: {importance:.4f}")

# Save ranked vegetables for each test sample
ranked_vegetables = pd.DataFrame(index=range(len(X_test)))
for i, veg in enumerate(vegetables):
    ranked_vegetables[f'Rank_{i+1}'] = normalized_predictions_df.apply(
        lambda row: vegetables[np.argsort(row.values)[::-1][i]], 
        axis=1
    )

# Add prediction confidence scores
for i in range(3):  # Save confidence scores for top 3 predictions
    ranked_vegetables[f'Confidence_{i+1}'] = normalized_predictions_df.apply(
        lambda row: round(sorted(row.values, reverse=True)[i] * 100, 1),
        axis=1
    )

# Add actual vegetable for comparison
ranked_vegetables['Actual'] = y_test.values

# Show sample predictions
print("\nSample Predictions (first 5 samples):")
sample_display = ranked_vegetables.iloc[:5, :7]  # Top 3 predictions with confidences + actual
print(sample_display)

# Save results to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ranked_vegetables.to_csv(f'results/vegetable_predictions_{timestamp}.csv', index=False)

# Save confusion matrix-like data
confusion = pd.DataFrame(0, index=vegetables, columns=vegetables)
for true_veg, pred_veg in zip(y_test, [vegetables[i] for i in pred_top1]):
    confusion.loc[true_veg, pred_veg] += 1

confusion.to_csv(f'results/confusion_matrix_{timestamp}.csv')

print(f"\nFull results saved to results/vegetable_predictions_{timestamp}.csv")
print(f"Confusion matrix saved to results/confusion_matrix_{timestamp}.csv")
print("\nTraining complete!")