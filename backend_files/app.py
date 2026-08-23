import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app with a name
sales_forecasting_api = Flask("SuperKart Sales Forecasting API")

# Define the assign_product_category function (re-using logic from notebook)
def assign_product_category(product_type):
    if product_type in ['Frozen Foods', 'Dairy', 'Fruits and Vegetables', 'Meat', 'Seafood']:
        return 'Perishable Food'
    elif product_type in ['Canned', 'Baking Goods', 'Starchy Foods', 'Snack Foods', 'Breakfast']:
        return 'Non-Perishable Food'
    elif product_type in ['Soft Drinks', 'Hard Drinks']:
        return 'Drinks'
    elif product_type in ['Health and Hygiene', 'Household']:
        return 'Non-Consumable'
    else:
        return 'Others'

# Load the trained sales forecasting model
# The model object is a scikit-learn Pipeline that includes preprocessing and the best estimator
model = joblib.load("SuperKart_sales_forecasting_model_v1_0.joblib")

# Define the final features expected by the model's pipeline after feature engineering
# This order must match the order the model was trained with
final_model_features = [
    'Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age_Years',
    'Product_Sugar_Content', 'Store_Size', 'Store_Location_City_Type', 'Store_Type',
    'Product_Id_Prefix', 'Product_Type_Category'
]

# Define a route for the home page
@sales_forecasting_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Forecasting API! Send POST requests to /v1/predict or /v1/predictbatch."

# Define an endpoint to predict sales for a single product
@sales_forecasting_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request, which contains original features
    product_data = request.get_json()

    # Convert the dictionary to a DataFrame for processing
    input_df = pd.DataFrame([product_data])

    # --- Re-apply Feature Engineering steps (as done in the notebook) ---
    # 1. Extract Product_Id_Prefix from Product_Id
    input_df['Product_Id_Prefix'] = input_df['Product_Id'].apply(lambda x: x[0:2])

    # 2. Calculate Store_Age_Years from Store_Establishment_Year
    # Use 2009 as the 'current_year' for consistency with notebook's training data derivation
    current_year_for_age = 2009
    input_df['Store_Age_Years'] = current_year_for_age - input_df['Store_Establishment_Year']

    # 3. Create Product_Type_Category by grouping Product_Type
    input_df['Product_Type_Category'] = input_df['Product_Type'].apply(assign_product_category)

    # Drop original columns that were transformed or are not part of the final model features
    input_df.drop(columns=['Product_Id', 'Store_Establishment_Year', 'Product_Type'], inplace=True)

    # Ensure the DataFrame has the correct features in the expected order
    processed_input_df = input_df[final_model_features]

    # Make a sales prediction using the trained model
    prediction = model.predict(processed_input_df).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Predicted_Sales': round(prediction, 2)})

# Define an endpoint to predict sales for a batch of products
@sales_forecasting_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a DataFrame
    input_df = pd.read_csv(file)

    # Store original Product_Ids if available and useful for output mapping
    # Assuming 'Product_Id' is present in the batch CSV for identification
    original_product_ids = None
    if 'Product_Id' in input_df.columns:
        original_product_ids = input_df['Product_Id'].values.tolist()

    # --- Re-apply Feature Engineering steps (as done in the notebook) ---
    # 1. Extract Product_Id_Prefix from Product_Id
    input_df['Product_Id_Prefix'] = input_df['Product_Id'].apply(lambda x: x[0:2])

    # 2. Calculate Store_Age_Years from Store_Establishment_Year
    current_year_for_age = 2026
    input_df['Store_Age_Years'] = current_year_for_age - input_df['Store_Establishment_Year']

    # 3. Create Product_Type_Category by grouping Product_Type
    input_df['Product_Type_Category'] = input_df['Product_Type'].apply(assign_product_category)

    # Drop original columns that were transformed or are not part of the final model features
    # Note: 'Store_Id' was also dropped during notebook feature engineering, so drop it here too if present in batch
    cols_to_drop = ['Product_Id', 'Store_Establishment_Year', 'Product_Type']
    if 'Store_Id' in input_df.columns:
        cols_to_drop.append('Store_Id')
    input_df.drop(columns=cols_to_drop, inplace=True)

    # Ensure the DataFrame has the correct features in the expected order
    processed_input_df = input_df[final_model_features]

    # Make predictions for the batch data
    predictions = model.predict(processed_input_df).tolist()

    # Return predictions, mapping to Product_Id if available, otherwise just the list of predictions
    if original_product_ids and len(original_product_ids) == len(predictions):
        output_dict = {pid: round(pred, 2) for pid, pred in zip(original_product_ids, predictions)}
        return jsonify({'Batch_Predictions': output_dict})
    else:
        return jsonify({'Batch_Predictions': [round(pred, 2) for pred in predictions]})

# Run the Flask app in debug mode
if __name__ == '__main__':
    # host='0.0.0.0' allows access from outside the container (e.g., from Codespaces port forwarding)
    # port=5000 is a common default for Flask, can be changed if needed
    sales_forecasting_api.run(debug=True, host='0.0.0.0', port=5000)
