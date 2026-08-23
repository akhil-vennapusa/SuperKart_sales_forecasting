import requests
import streamlit as st
import pandas as pd

st.title("SuperKart Sales Forecasting")

# --- Online Prediction --- #
st.subheader("Online Sales Prediction")

# Input fields for SuperKart product and store data
product_id = st.text_input("Product ID (e.g., FD6114, DR1234)", value="FD6114")
product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66, step=0.01)
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_allocated_area = st.number_input("Product Allocated Area (ratio)", min_value=0.0, max_value=1.0, value=0.027, step=0.001, format="%.3f")
product_mrp = st.number_input("Product MRP (Maximum Retail Price)", min_value=0.0, value=117.08, step=0.01)
store_establishment_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2024, value=2009)
store_size = st.selectbox("Store Size", ["Medium", "High", "Small"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Supermarket Type1", "Departmental Store", "Supermarket Type2", "Food Mart"])
product_type = st.selectbox("Product Type", [
    'Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene', 
    'Snack Foods', 'Soft Drinks', 'Fruits and Vegetables', 'Household', 
    'Hard Drinks', 'Others', 'Meat', 'Starchy Foods', 'Breakfast', 'Seafood'
])

# Create a dictionary with the input data
product_data = {
    'Product_Id': product_id,
    'Product_Weight': product_weight,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Allocated_Area': product_allocated_area,
    'Product_MRP': product_mrp,
    'Store_Establishment_Year': store_establishment_year,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type,
    'Product_Type': product_type
}

# backend url
backend_url_online = "https://backend:7860/v1/predict" # Ensure this points to your Flask backend URL

if st.button("Predict Sales", type='primary'):
    try:
        response = requests.post(backend_url_online, json=product_data)
        if response.status_code == 200:
            result = response.json()
            predicted_sales = result.get("Predicted_Sales")
            if predicted_sales is not None:
                st.success(f"The predicted sales for Product ID **{product_id}** is **${predicted_sales:,.2f}**.")
            else:
                st.error("Prediction result not found in the response.")
        else:
            st.error(f"Error in API request: Status Code {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the backend API. Please ensure the backend is running and accessible.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# --- Batch Prediction --- #
st.subheader("Batch Sales Prediction")

file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])


backend_url_batch = "https://backend:7860/v1/predictbatch" # Ensure this points to your Flask backend URL

if file is not None:
    if st.button("Predict Sales for Batch", type='secondary'):
        try:
            files = {'file': (file.name, file.getvalue(), 'text/csv')}
            response = requests.post(backend_url_batch, files=files)

            if response.status_code == 200:
                result = response.json()
                st.header("Batch Prediction Results")
                
                # Convert dictionary result to DataFrame for better display
                if 'Batch_Predictions' in result and isinstance(result['Batch_Predictions'], dict):
                    df_results = pd.DataFrame(list(result['Batch_Predictions'].items()), columns=['Product_Id', 'Predicted_Sales'])
                    st.dataframe(df_results.set_index('Product_Id'))
                elif 'Batch_Predictions' in result and isinstance(result['Batch_Predictions'], list):
                    st.write("Predictions returned as a list:")
                    st.write(result['Batch_Predictions'])
                else:
                    st.json(result)
            else:
                st.error(f"Error in batch API request: Status Code {response.status_code} - {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not connect to the backend API. Please ensure the backend is running and accessible.")
        except Exception as e:
            st.error(f"An unexpected error occurred during batch prediction: {e}")
