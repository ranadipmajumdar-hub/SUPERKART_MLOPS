
import streamlit as st
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SuperKart Sales Forecast",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# HUGGING FACE MODEL CONFIGURATION
# ============================================================

MODEL_REPO = (
    "ranadipmajumdar/"
    "Superkart_MLOps-model"
)

MODEL_FILENAME = (
    "superkart_sales_model.joblib"
)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    model = joblib.load(
        model_path
    )

    return model


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "📊 SuperKart Sales Forecasting"
)

st.write(
    """
    This application predicts the expected sales revenue
    for a product in a particular store using the trained
    SuperKart machine learning model.
    """
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

    st.success(
        "Sales forecasting model loaded successfully."
    )

except Exception as e:

    st.error(
        "Unable to load the forecasting model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader(
    "Enter Product and Store Details"
)

col1, col2 = st.columns(2)


# ============================================================
# PRODUCT INFORMATION
# ============================================================

with col1:

    product_weight = st.number_input(
        "Product Weight",
        min_value=0.0,
        value=12.0,
        step=0.1
    )

    product_sugar_content = st.selectbox(
        "Product Sugar Content",
        [
            "Low Sugar",
            "Regular",
            "No Sugar"
        ]
    )

    product_allocated_area = st.number_input(
        "Product Allocated Area",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01
    )

    product_type = st.selectbox(
        "Product Type",
        [
            "Meat",
            "Snack Foods",
            "Hard Drinks",
            "Dairy",
            "Canned",
            "Soft Drinks",
            "Health and Hygiene",
            "Baking Goods",
            "Bread",
            "Breakfast",
            "Frozen Foods",
            "Fruits and Vegetables",
            "Household",
            "Seafood",
            "Starchy Foods",
            "Others"
        ]
    )

    product_mrp = st.number_input(
        "Product MRP",
        min_value=0.0,
        value=150.0,
        step=1.0
    )


# ============================================================
# STORE INFORMATION
# ============================================================

with col2:

    store_establishment_year = st.number_input(
        "Store Establishment Year",
        min_value=1980,
        max_value=2030,
        value=2000,
        step=1
    )

    store_size = st.selectbox(
        "Store Size",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    store_location_city_type = st.selectbox(
        "Store Location City Type",
        [
            "Tier 1",
            "Tier 2",
            "Tier 3"
        ]
    )

    store_type = st.selectbox(
        "Store Type",
        [
            "Departmental Store",
            "Supermarket Type 1",
            "Supermarket Type 2",
            "Food Mart"
        ]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Sales",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAKE PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create input dictionary
    # --------------------------------------------------------

    input_data = {

        "Product_Weight": [
            product_weight
        ],

        "Product_Sugar_Content": [
            product_sugar_content
        ],

        "Product_Allocated_Area": [
            product_allocated_area
        ],

        "Product_Type": [
            product_type
        ],

        "Product_MRP": [
            product_mrp
        ],

        "Store_Establishment_Year": [
            store_establishment_year
        ],

        "Store_Size": [
            store_size
        ],

        "Store_Location_City_Type": [
            store_location_city_type
        ],

        "Store_Type": [
            store_type
        ]
    }


    # --------------------------------------------------------
    # Convert inputs into DataFrame
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        input_data
    )


    # --------------------------------------------------------
    # Display input DataFrame
    # --------------------------------------------------------

    st.subheader(
        "Input Data"
    )

    st.dataframe(
        input_df,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    try:

        prediction = model.predict(
            input_df
        )

        predicted_sales = float(
            prediction[0]
        )


        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.subheader(
            "Predicted Sales Revenue"
        )

        st.metric(
            label="Estimated Product Store Sales",
            value=f"{predicted_sales:,.2f}"
        )

        st.success(
            "Sales prediction generated successfully."
        )


    except Exception as e:

        st.error(
            "An error occurred while generating "
            "the prediction."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SuperKart Sales Forecasting | "
    "MLOps Project"
)
