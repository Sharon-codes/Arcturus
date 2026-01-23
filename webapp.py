import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

"""
Flask Web Application for Meteorite Organics Spectral Analysis
LIGHTWEIGHT DEMO MODE - No PyTorch/ML Dependencies
"""

app = Flask(__name__)
CORS(app)

# --- MOCK DATA GENERATOR ---
def generate_mock_spectrum():
    # Generate common X axis (Wavenumbers) reversed like standard IR
    x = np.linspace(4000, 500, 2048)
    
    # Generate Synthetic Spectrum (Y) - A nice N-PAH signature
    # Baseline
    y = np.random.normal(0.02, 0.005, 2048) + 0.05
    
    # 3050 cm-1 (C-H Str)
    y += 0.8 * np.exp(-((x - 3050)**2) / (2 * 40**2))
    # 2200 cm-1 (C-N Nitrile - The "Life" Signal)
    y += 0.6 * np.exp(-((x - 2200)**2) / (2 * 20**2)) 
    # 1600 cm-1 (C=C Skeletal)
    y += 0.5 * np.exp(-((x - 1600)**2) / (2 * 30**2))
    # 1200 cm-1 (Fingerprint)
    y += 0.4 * np.exp(-((x - 1200)**2) / (2 * 50**2))

    # Generate Synthetic Saliency (Red Line) - Focused on the "Life" peaks
    saliency = np.zeros_like(x)
    # AI focuses heavily on the Nitrile peak (Biological precursor)
    saliency += 0.9 * np.exp(-((x - 2200)**2) / (2 * 15**2))
    # And the C-H stretch
    saliency += 0.5 * np.exp(-((x - 3050)**2) / (2 * 30**2)) 
    
    # Subsample for JSON transfer
    return x.tolist()[::5], y.tolist()[::5], saliency.tolist()[::5]

# Generate static data once
DEMO_X, DEMO_Y, DEMO_SALIENCY = generate_mock_spectrum()

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    # Hardcoded stats for the demo
    return jsonify({
        'total_spectra': 14205,
        'pahdb_count': 3500,
        'pds_count': 10705,
        'clusters': {0: 3400, 1: 8200, 2: 1500},
        'families': {
            0: 'N-PAH', 
            1: 'Pure PAH', 
            2: 'Aliphatic', 
            3: 'O-PAH', 
            4: 'Methyl-PAH'
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict_spectrum():
    """
    MOCK PREDICTION ENDPOINT
    Returns a pre-calculated result for 'Nitrogen-PAH' to demonstrate the UI features.
    No heavy model loading occurs.
    """
    # Verify file text is present (for UI completeness)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    filename = file.filename if file.filename else "demo_sample.tab"
    
    # Return the "Perfect" Demo Result
    result = {
        'filename': filename,
        'predicted_family': 'N-PAH', # The most interesting class (Biotic precursor)
        'family_id': 0, # Assuming N-PAH is ID 0 or mapped correctly locally
        'confidence': 0.985, # High confidence for the demo
        'top_peak_wavenumber': 2200.5, # The specific Nitrile peak
        'spectrum': {
            'x': DEMO_X,
            'y': DEMO_Y,
            'saliency': DEMO_SALIENCY
        },
        'explanation': {
            'what_is_saliency': 'Shows which parts of your spectrum the AI considered most important',
            'confidence_interpretation': '98.5% confident - Strong Match',
            'family_meaning': 'N-PAH - Contains nitrogen, potential DNA/RNA building blocks'
        }
    }
    
    return jsonify(result)

@app.route('/api/search')
def search_spectra():
    return jsonify([])

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting CHANDRA Web Application (LIGHTWEIGHT MODE)")
    print("="*60)
    print("\n📊 Access the dashboard at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
