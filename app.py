from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
MODEL_DIR = 'models'
DATA_PATH = 'data/optimized_dataset.csv'

# Load the dataset for reference data and feature engineering
data = pd.read_csv(DATA_PATH)

# Create crop rotation compatibility matrix
# Higher value = better rotation pair, 0 = poor rotation choice
# This is a simplified example - you would need to replace with actual crop rotation rules
def create_rotation_matrix(vegetables):
    # Create a default matrix with neutral values
    n_vegetables = len(vegetables)
    rotation_matrix = pd.DataFrame(0.5, index=vegetables, columns=vegetables)
    
    # Set diagonal to 0 (planting same crop twice is bad)
    np.fill_diagonal(rotation_matrix.values, 0)
    
    # Define good rotation pairs based on crop rotation principles
    good_rotations = {
        'Potato': ['Beans', 'Cabbage', 'Snake Gourd', 'Pumpkin'],
        'Tomato': ['Beans', 'Cabbage', 'Carrot'],
        'Brinjal': ['Beans', 'Cabbage', 'Carrot'],
        'Green Chilli': ['Beans', 'Carrot', 'Cabbage'],
        'Beans': ['Cabbage', 'Pumpkin', 'Snake Gourd', 'Carrot', 'Tomato'],
        'Cabbage': ['Carrot', 'Potato', 'Pumpkin', 'Snake Gourd'],
        'Carrot': ['Tomato', 'Brinjal', 'Green Chilli', 'Pumpkin'],
        'Snake Gourd': ['Beans', 'Cabbage', 'Carrot'],
        'Pumpkin': ['Beans', 'Cabbage', 'Carrot'],
        'Lime': []  # Lime is perennial; generally not rotated in annual systems
    }
    
    # Set the good rotation values (0.8 = good rotation pair)
    for previous, next_crops in good_rotations.items():
        if previous in vegetables:
            for next_crop in next_crops:
                if next_crop in vegetables:
                    rotation_matrix.loc[previous, next_crop] = 0.8
    
    # Bad rotation pairs (0.1 = bad rotation), especially same family or same type
    bad_rotations = {
        'Potato': ['Tomato', 'Brinjal', 'Green Chilli'],  # Solanaceae family
        'Tomato': ['Potato', 'Brinjal', 'Green Chilli'],
        'Brinjal': ['Tomato', 'Potato', 'Green Chilli'],
        'Green Chilli': ['Potato', 'Tomato', 'Brinjal'],
        'Cabbage': ['Cabbage'],  # Brassicas deplete soil and attract pests
        'Pumpkin': ['Snake Gourd', 'Pumpkin'],  # Same family, heavy feeders
        'Snake Gourd': ['Pumpkin', 'Snake Gourd'],
        'Lime': []  # Not applicable, but avoid planting under lime
    }
    
    # Set the bad rotation values
    for previous, next_crops in bad_rotations.items():
        if previous in vegetables:
            for next_crop in next_crops:
                if next_crop in vegetables:
                    rotation_matrix.loc[previous, next_crop] = 0.1
                    
    return rotation_matrix

# Create a table for seasonal suitability by month (1-12)
# Higher value = better suitability for planting in that month
def create_seasonal_matrix(vegetables):
    months = range(1, 13)  # Jan (1) to Dec (12)
    seasonal_matrix = pd.DataFrame(0.5, index=vegetables, columns=months)
    
    # Maha Season: October (10) to February (2)
    maha_months = [10, 11, 12, 1, 2]
    # Yala Season: May (5) to August (8)
    yala_months = [5, 6, 7, 8]
    # Inter-monsoon: March-April (3, 4) & September (9)
    inter_months = [3, 4, 9]
    
    # Crops that perform best in Maha
    maha_best = ['Potato', 'Beans', 'Cabbage', 'Pumpkin', 'Snake Gourd', 'Carrot']
    for crop in maha_best:
        if crop in vegetables:
            seasonal_matrix.loc[crop, maha_months] = 0.9
            seasonal_matrix.loc[crop, yala_months] = 0.7
            seasonal_matrix.loc[crop, inter_months] = 0.7
    
    # Crops that perform best in Yala
    yala_best = ['Tomato', 'Brinjal', 'Green Chilli']
    for crop in yala_best:
        if crop in vegetables:
            seasonal_matrix.loc[crop, yala_months] = 0.9
            seasonal_matrix.loc[crop, maha_months] = 0.7
            seasonal_matrix.loc[crop, inter_months] = 0.7
    
    # Lime is perennial, productive year-round
    if 'Lime' in vegetables:
        seasonal_matrix.loc['Lime', :] = 0.9
    
    return seasonal_matrix

# Calculate days to harvest from planting month to target harvest month
def calculate_growing_period(planting_month, harvest_month):
    if harvest_month >= planting_month:
        return harvest_month - planting_month
    else:
        return (12 - planting_month) + harvest_month  # Wrap around for next year

# Function to check if a crop can reach maturity in the given time period
def maturity_adjustment(crop, months_to_harvest):
    # Maturity times in months for your 10 crops (Sri Lanka, hill country)
    maturity_table = {
        'Potato': 3.5,
        'Tomato': 3.5,
        'Beans': 2.5,
        'Cabbage': 3.5,
        'Brinjal': 4,
        'Snake Gourd': 3,
        'Lime': 6,        # Starts yielding in 6 months, but fruits year-round after
        'Pumpkin': 4,
        'Carrot': 3,
        'Green Chilli': 4
    }
    
    # Default to 4 months if crop not in table (though all your crops are listed)
    required_months = maturity_table.get(crop, 4)
    
    # Special case: Lime is perennial, cycles after maturity
    if crop == 'Lime':
        # After 6 months, it's productive continuously
        if months_to_harvest >= 6:
            return 1.0
        elif months_to_harvest >= 4:
            return 0.5
        else:
            return 0.1
    
    # If enough time to mature, full score
    if months_to_harvest >= required_months:
        return 1.0
    # If reasonably close, partial score
    elif months_to_harvest >= required_months * 0.75:
        return 0.5
    # If too short, penalize
    else:
        return 0.1


# Load all trained models
def load_models():
    models = {}
    vegetables = []
    
    for filename in os.listdir(MODEL_DIR):
        if filename.startswith('cb_model_') and filename.endswith('.cbm'):
            veg_name = filename[9:-4]  # Extract vegetable name from filename
            model_path = os.path.join(MODEL_DIR, filename)
            models[veg_name] = CatBoostRegressor().load_model(model_path)
            vegetables.append(veg_name)
    
    return models, vegetables

# Initialize models and resources
models, vegetables = load_models()
rotation_matrix = create_rotation_matrix(vegetables)
seasonal_matrix = create_seasonal_matrix(vegetables)

# Feature preprocessing components - fitted on the original data
categorical_features = ['Month']
numerical_features = ['Avg_Price', 'Price_Std', 'Data_Count', 'Price_to_Count_Ratio', 'Month_Sin', 'Month_Cos']

data['Price_to_Count_Ratio'] = data['Avg_Price'] / (data['Data_Count'] + 1)
data['Month_Sin'] = np.sin(2 * np.pi * data['Month']/12)
data['Month_Cos'] = np.cos(2 * np.pi * data['Month']/12)

imputer = SimpleImputer(strategy='median')
imputer.fit(data[numerical_features])

scaler = StandardScaler()
scaler.fit(data[numerical_features])

@app.route('/api/vegetables', methods=['GET'])
def get_vegetables():
    """Return list of available vegetables"""
    return jsonify({
        'vegetables': vegetables
    })

@app.route('/api/recommend', methods=['POST'])
def recommend_crops():
    """
    Recommend crops based on:
    1. Target harvest month
    2. Previous crop (for rotation)
    3. Desired crop to plant (optional - to check compatibility)
    """
    try:
        # Parse request data
        request_data = request.json
        target_month = int(request_data.get('targetMonth', 0))
        previous_crop = request_data.get('previousCrop', None)
        desired_crop = request_data.get('desiredCrop', None)
        planting_month = int(request_data.get('plantingMonth', datetime.now().month))
        
        # Validate input
        if not (1 <= target_month <= 12):
            return jsonify({'error': 'Invalid target month. Must be between 1 and 12.'}), 400
        
        if previous_crop and previous_crop not in vegetables:
            return jsonify({'error': f'Unknown previous crop: {previous_crop}. Available crops: {", ".join(vegetables)}'}), 400
            
        if desired_crop and desired_crop not in vegetables:
            return jsonify({'error': f'Unknown desired crop: {desired_crop}. Available crops: {", ".join(vegetables)}'}), 400
        
        # Prepare input features
        month_sin = np.sin(2 * np.pi * target_month/12)
        month_cos = np.cos(2 * np.pi * target_month/12)
        
        # Use median values for price-related features as placeholders
        # In a real system, you might want to use actual market data or forecasts
        input_df = pd.DataFrame({
            'Month': [target_month],
            'Avg_Price': [data['Avg_Price'].median()],
            'Price_Std': [data['Price_Std'].median()], 
            'Data_Count': [data['Data_Count'].median()],
            'Price_to_Count_Ratio': [data['Price_to_Count_Ratio'].median()],
            'Month_Sin': [month_sin],
            'Month_Cos': [month_cos]
        })
        
        # Preprocess input features
        input_df[numerical_features] = imputer.transform(input_df[numerical_features])
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        # Calculate months available for growing
        months_to_harvest = calculate_growing_period(planting_month, target_month)
        
        # Generate base predictions
        predictions = {}
        for veg in vegetables:
            # Get base prediction from model
            base_score = models[veg].predict(input_df)[0]
            
            # Normalize to 0-1 scale if not already
            base_score = max(0, min(1, base_score))
            
            # Apply crop rotation adjustment if previous crop is specified
            rotation_factor = 1.0
            if previous_crop:
                rotation_factor = rotation_matrix.loc[previous_crop, veg]
            
            # Apply seasonal planting adjustment
            seasonal_factor = seasonal_matrix.loc[veg, planting_month]
            
            # Apply maturity adjustment
            maturity_factor = maturity_adjustment(veg, months_to_harvest)
            
            # Combine all factors
            # Base score is weighted most heavily
            final_score = (base_score * 0.5) + \
                         (rotation_factor * 0.2) + \
                         (seasonal_factor * 0.2) + \
                         (maturity_factor * 0.1)
            
            # Store both combined and individual scores for transparency
            predictions[veg] = {
                'finalScore': round(final_score * 100, 1),  # Convert to percentage
                'baseScore': round(base_score * 100, 1),
                'rotationScore': round(rotation_factor * 100, 1),
                'seasonalScore': round(seasonal_factor * 100, 1),
                'maturityScore': round(maturity_factor * 100, 1),
                'monthsToHarvest': months_to_harvest,
                # Add descriptive info for user feedback
                'rotationComment': get_rotation_comment(previous_crop, veg, rotation_factor) if previous_crop else None,
                'seasonalComment': get_seasonal_comment(veg, planting_month, seasonal_factor),
                'maturityComment': get_maturity_comment(veg, months_to_harvest, maturity_factor)
            }
        
        # Sort predictions by final score
        sorted_predictions = [
            {'crop': veg, **scores} 
            for veg, scores in sorted(predictions.items(), key=lambda x: x[1]['finalScore'], reverse=True)
        ]
        
        # Return all predictions for transparency
        return jsonify({
            'recommendations': sorted_predictions,
            'plantingMonth': planting_month,
            'harvestMonth': target_month,
            'monthsToHarvest': months_to_harvest,
            'previousCrop': previous_crop
        })
        
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Helper functions for generating user-friendly comments
def get_rotation_comment(previous_crop, current_crop, score):
    if score >= 0.75:
        return f"{current_crop} is an excellent rotation choice after {previous_crop}."
    elif score >= 0.5:
        return f"{current_crop} is a reasonable rotation choice after {previous_crop}."
    elif score >= 0.25:
        return f"{current_crop} is not ideal for rotation after {previous_crop}."
    else:
        return f"Avoid planting {current_crop} after {previous_crop} - they are in the same family or have incompatible needs."

def get_seasonal_comment(crop, month, score):
    month_names = ["", "January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    
    if score >= 0.8:
        return f"{month_names[month]} is an ideal time to plant {crop}."
    elif score >= 0.6:
        return f"{month_names[month]} is a good time to plant {crop}."
    elif score >= 0.4:
        return f"{month_names[month]} is an acceptable time to plant {crop}, but not optimal."
    else:
        return f"{month_names[month]} is not recommended for planting {crop}."

def get_maturity_comment(crop, months_available, score):
    if score >= 0.9:
        return f"{crop} will have plenty of time to mature before harvest."
    elif score >= 0.5:
        return f"{crop} should have just enough time to mature before harvest."
    else:
        return f"{crop} may not have enough time to fully mature before the target harvest month."

if __name__ == '__main__':
    app.run(debug=True, port=6000)