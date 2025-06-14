from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('house_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Updated area price multipliers for Pune (2024 market rates)
# Base price is multiplied by these factors to get current market rates
AREA_MULTIPLIERS = {
    'koregaon_park': 4.5,    # Premium area (₹15,000-20,000 per sq ft)
    'kalyani_nagar': 4.0,    # High-end area (₹12,000-18,000 per sq ft)
    'viman_nagar': 3.8,      # Upscale area (₹11,000-16,000 per sq ft)
    'baner': 3.5,            # Developing premium area (₹10,000-15,000 per sq ft)
    'wakad': 3.0,            # IT hub (₹8,000-12,000 per sq ft)
    'hinjewadi': 2.8,        # IT hub (₹7,000-11,000 per sq ft)
    'kharadi': 3.2,          # Developing area (₹9,000-13,000 per sq ft)
    'aundh': 3.3,            # Established area (₹9,000-14,000 per sq ft)
    'pashan': 2.9,           # Residential area (₹8,000-12,000 per sq ft)
    'magarpatta': 3.6        # Planned township (₹10,000-15,000 per sq ft)
}

# Property type multipliers
PROPERTY_TYPE_MULTIPLIERS = {
    'apartment': 1.0,        # Base price
    'villa': 1.8,           # Villas are typically more expensive
    'plot': 1.5,            # Plots have higher value
    'penthouse': 2.0,       # Penthouses are premium
    'row_house': 1.2        # Row houses are slightly more expensive than apartments
}

# Furnishing status multipliers
FURNISHING_MULTIPLIERS = {
    'unfurnished': 1.0,     # Base price
    'semi_furnished': 1.1,  # 10% more
    'fully_furnished': 1.25 # 25% more
}

# Base price adjustment factor to account for overall market rates
BASE_PRICE_ADJUSTMENT = 2.5  # This will multiply the base prediction before area multiplier

def calculate_feature_multipliers(form_data):
    """Calculate multipliers based on property features"""
    multipliers = 1.0
    
    try:
        # Bedroom multiplier (more bedrooms = higher price)
        bedroom_mult = 1.0 + (int(form_data.get('bedrooms', 1)) - 1) * 0.15
        multipliers *= bedroom_mult
        
        # Bathroom multiplier
        bathroom_mult = 1.0 + (int(form_data.get('bathrooms', 1)) - 1) * 0.1
        multipliers *= bathroom_mult
        
        # Floor multiplier (higher floors = higher price, but with diminishing returns)
        floor = int(form_data.get('floor', 0))
        if floor > 0:
            floor_mult = 1.0 + (min(floor, 10) * 0.02)  # 2% increase per floor up to 10th floor
            multipliers *= floor_mult
        
        # Age multiplier (newer properties = higher price)
        age = int(form_data.get('age', 0))
        age_mult = 1.0 - (min(age, 30) * 0.01)  # 1% decrease per year up to 30 years
        multipliers *= max(age_mult, 0.7)  # Minimum 70% of base price
        
        # Parking multiplier
        parking = int(form_data.get('parking', 0))
        parking_mult = 1.0 + (parking * 0.05)  # 5% increase per parking spot
        multipliers *= parking_mult
        
    except (ValueError, TypeError) as e:
        print(f"Error in calculate_feature_multipliers: {e}")
        return 1.0  # Return default multiplier if there's an error
        
    return multipliers

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all form data
        form_data = request.form.to_dict()
        
        # Validate required fields
        required_fields = ['property_type', 'area', 'total_area', 'bedrooms', 
                         'bathrooms', 'floor', 'age', 'furnishing', 'parking', 'overallqual']
        
        for field in required_fields:
            if field not in form_data or not form_data[field]:
                return render_template('index.html', 
                                     prediction_text=f"❌ Error: Please fill in all required fields. Missing: {field}")
        
        # Convert and validate numeric fields
        try:
            total_area = float(form_data['total_area'])
            overall_qual = float(form_data['overallqual'])
            if total_area <= 0 or overall_qual < 1 or overall_qual > 10:
                raise ValueError("Invalid numeric values")
        except ValueError as e:
            return render_template('index.html', 
                                 prediction_text="❌ Error: Please enter valid numbers for area and quality")
        
        # Calculate base price using the model
        # Using total_area and overall_qual as main features
        features = np.array([[total_area, overall_qual, 1, 0]])  # Using default values for unused features
        features_scaled = scaler.transform(features)
        base_prediction = model.predict(features_scaled)[0]
        
        # Apply all multipliers
        base_price = base_prediction * BASE_PRICE_ADJUSTMENT
        
        # Apply property type multiplier
        property_type = form_data['property_type']
        property_multiplier = PROPERTY_TYPE_MULTIPLIERS.get(property_type, 1.0)
        base_price *= property_multiplier
        
        # Apply area multiplier
        area = form_data['area']
        area_multiplier = AREA_MULTIPLIERS.get(area, 1.0)
        base_price *= area_multiplier
        
        # Apply furnishing multiplier
        furnishing = form_data['furnishing']
        furnishing_multiplier = FURNISHING_MULTIPLIERS.get(furnishing, 1.0)
        base_price *= furnishing_multiplier
        
        # Apply feature-based multipliers
        feature_multipliers = calculate_feature_multipliers(form_data)
        final_prediction = base_price * feature_multipliers
        
        # Calculate price per square foot
        price_per_sqft = int(final_prediction / total_area)
        
        # Format the output
        area_display = area.replace('_', ' ').title()
        property_type_display = property_type.replace('_', ' ').title()
        
        prediction_text = (
            f"Estimated Price for {property_type_display} in {area_display}: "
            f"₹{int(final_prediction):,} (₹{price_per_sqft:,} per sq ft)\n"
            f"Property Details: {form_data['bedrooms']} BHK, {form_data['bathrooms']} Bathrooms, "
            f"Floor {form_data['floor']}, {form_data['furnishing'].replace('_', ' ').title()}, "
            f"{form_data['parking']} Parking"
        )
        
        return render_template('index.html', 
                             prediction_text=prediction_text,
                             selected_area=area,
                             selected_property_type=property_type)
                             
    except Exception as e:
        print(f"Error in predict route: {str(e)}")  # For debugging
        return render_template('index.html', 
                             prediction_text=f"❌ Error: Please check your input values. Details: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
