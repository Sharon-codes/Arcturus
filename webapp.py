"""
Flask Web Application for Meteorite Organics Spectral Analysis
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.model import SpectralCNN
from src.data_ingestion import UniversalParser
from src.preprocessing import SpectralPreprocessor
from src.utils import FAMILIES
from src.discovery import ChemicalDiscovery

app = Flask(__name__)
CORS(app)

# Global model and data
MODEL = None
DEVICE = None
RESULTS_DF = None
DISCOVERY = None
PARSER = None
PREPROCESSOR = None

def load_model_and_data():
    """Load trained model and results on startup"""
    global MODEL, DEVICE, RESULTS_DF, DISCOVERY, PARSER, PREPROCESSOR
    
    print("Loading model and data...")
    
    # Load model
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL = SpectralCNN(num_classes=len(FAMILIES))
    MODEL.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()
    
    # Load results
    RESULTS_DF = pd.read_csv('hybrid_scientific_results.csv')
    
    # Initialize discovery and preprocessing
    DISCOVERY = ChemicalDiscovery(MODEL, device=DEVICE)
    PARSER = UniversalParser()
    PREPROCESSOR = SpectralPreprocessor()
    
    print(f"✓ Model loaded on {DEVICE}")
    print(f"✓ Results loaded: {len(RESULTS_DF)} spectra")

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Get dataset statistics"""
    pds = RESULTS_DF[RESULTS_DF['source'] == 'PDS']
    pahdb = RESULTS_DF[RESULTS_DF['source'] == 'PAHdb']
    
    cluster_counts = pds['cluster_membership'].value_counts().to_dict()
    
    stats = {
        'total_spectra': len(RESULTS_DF),
        'pahdb_count': len(pahdb),
        'pds_count': len(pds),
        'clusters': {
            int(k): int(v) for k, v in cluster_counts.items() if k != -1
        },
        'families': FAMILIES
    }
    
    return jsonify(stats)

@app.route('/api/cluster/<int:cluster_id>')
def get_cluster_details(cluster_id):
    """Get details for a specific cluster"""
    pds = RESULTS_DF[RESULTS_DF['source'] == 'PDS']
    cluster_data = pds[pds['cluster_membership'] == cluster_id]
    
    result = {
        'cluster_id': cluster_id,
        'count': len(cluster_data),
        'percentage': (len(cluster_data) / len(pds)) * 100,
        'samples': cluster_data['id'].head(20).tolist()
    }
    
    return jsonify(result)

@app.route('/api/predict', methods=['POST'])
def predict_spectrum():
    """Predict chemical family for uploaded spectrum"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Save temporarily
        temp_path = Path('temp_upload') / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        file.save(str(temp_path))
        
        # Parse spectrum
        try:
            if file.filename.endswith('.tab'):
                spectrum = PARSER.parse_pds_tab(str(temp_path))
            else:
                return jsonify({'error': 'Unsupported file format. Use .tab files'}), 400
            
            # Preprocess
            x_proc, y_proc = PREPROCESSOR.full_pipeline(spectrum.x, spectrum.y)
            
            # Interpolate to 2048 points
            from scipy.interpolate import interp1d
            common_x = np.linspace(spectrum.x.min(), spectrum.x.max(), 2048)
            interpolator = interp1d(x_proc, y_proc, bounds_error=False, fill_value=0)
            y_interp = interpolator(common_x)
            
            # Predict
            with torch.no_grad():
                x_tensor = torch.FloatTensor(y_interp).unsqueeze(0).to(DEVICE)
                logits, embeddings = MODEL(x_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred = probs.max(1)
            
            # Get family
            family_id = pred.item()
            family_name = FAMILIES[family_id]
            confidence = conf.item()
            
            # Generate saliency map
            saliency = DISCOVERY.generate_saliency_map(y_interp, target_class=family_id)
            top_peak_idx = np.argmax(saliency)
            top_peak_wavenumber = common_x[top_peak_idx]
            
            result = {
                'filename': file.filename,
                'predicted_family': family_name,
                'family_id': family_id,
                'confidence': float(confidence),
                'top_peak_wavenumber': float(top_peak_wavenumber),
                'spectrum': {
                    'x': common_x.tolist()[::10],  # Subsample for transfer
                    'y': y_interp.tolist()[::10],
                    'saliency': saliency.tolist()[::10]
                }
            }
            
            # Cleanup
            temp_path.unlink()
            
            return jsonify(result)
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            return jsonify({'error': f'Error parsing file: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search')
def search_spectra():
    """Search for spectra by ID"""
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    matches = RESULTS_DF[RESULTS_DF['id'].str.lower().str.contains(query)]
    results = matches[['id', 'source', 'cluster_membership']].head(50).to_dict('records')
    
    return jsonify(results)

if __name__ == '__main__':
    load_model_and_data()
    print("\n" + "="*60)
    print("🚀 Starting Meteorite Organics Web Application")
    print("="*60)
    print("\n📊 Access the dashboard at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
