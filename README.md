# Pune House Price Predictor

A web application that predicts house prices in Pune based on various property features. The application uses machine learning to provide accurate price estimates for properties across different areas in Pune.

## Features

- Predicts house prices for different areas in Pune
- Considers multiple property features:
  - Property Type (Apartment, Villa, Plot, etc.)
  - Location (Koregaon Park, Kalyani Nagar, etc.)
  - Total Area
  - Number of Bedrooms (BHK)
  - Number of Bathrooms
  - Floor Number
  - Property Age
  - Furnishing Status
  - Parking Availability
  - Overall Quality Rating

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pune-house-price-predictor.git
cd pune-house-price-predictor
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and go to:
```
http://localhost:5000
```

3. Fill in the property details in the form and click "Calculate Price" to get the price prediction.

## Project Structure

```
pune-house-price-predictor/
├── app.py              # Flask application
├── models.py           # Model training code
├── templates/          # HTML templates
│   └── index.html     # Main form template
├── static/            # Static files (CSS, JS)
├── requirements.txt   # Python dependencies
├── README.md         # Project documentation
└── .gitignore        # Git ignore file
```

## Dependencies

- Flask
- scikit-learn
- numpy
- pandas
- pickle

## Note

The model files (`house_model.pkl` and `scaler.pkl`) are not included in the repository as they are generated during model training. You'll need to train the model first using the `models.py` script.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 