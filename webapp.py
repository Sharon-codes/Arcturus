import psutil
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="scipy.sparse")

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

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

# Global model and data - LAZY LOADING to save memory
MODEL = None
DEVICE = None
RESULTS_DF = None
DISCOVERY = None
PARSER = None
PREPROCESSOR = None

def load_model_lazy():
    """Load model only when needed"""
    global MODEL, DEVICE
    if MODEL is None:
        print("Loading model...")
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        MODEL = SpectralCNN(num_classes=len(FAMILIES))
        MODEL.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
        MODEL.to(DEVICE)
        MODEL.eval()
        mem_usage = get_memory_usage()
        print(f"✓ Model loaded on {DEVICE} (Memory: {mem_usage:.1f} MB)")
    return MODEL, DEVICE

def load_results_lazy():
    """Load results DataFrame only when needed"""
    global RESULTS_DF
    if RESULTS_DF is None:
        print("Loading results data...")
        # Load with memory optimization - only load necessary columns
        RESULTS_DF = pd.read_csv('hybrid_scientific_results.csv', usecols=['id', 'source', 'cluster_membership'])
        mem_usage = get_memory_usage()
        print(f"✓ Results loaded: {len(RESULTS_DF)} spectra (Memory: {mem_usage:.1f} MB)")
    return RESULTS_DF

def load_discovery_lazy():
    """Load discovery module only when needed"""
    global DISCOVERY, MODEL, DEVICE
    if DISCOVERY is None:
        if MODEL is None:
            load_model_lazy()
        DISCOVERY = ChemicalDiscovery(MODEL, device=DEVICE)
        print("✓ Discovery module loaded")
    return DISCOVERY

def load_parser_lazy():
    """Load parser only when needed"""
    global PARSER
    if PARSER is None:
        PARSER = UniversalParser()
        print("✓ Parser loaded")
    return PARSER

def load_preprocessor_lazy():
    """Load preprocessor only when needed"""
    global PREPROCESSOR
    if PREPROCESSOR is None:
        PREPROCESSOR = SpectralPreprocessor()
        print("✓ Preprocessor loaded")
    return PREPROCESSOR

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Get dataset statistics"""
    ensure_data_loaded()
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
    ensure_data_loaded()
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
        # Load required components lazily
        MODEL, DEVICE = load_model_lazy()
        PARSER = load_parser_lazy()
        PREPROCESSOR = load_preprocessor_lazy()
        DISCOVERY = load_discovery_lazy()
        
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
            
            # Memory cleanup
            del x_tensor, logits, embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Get family
            family_id = pred.item()
            family_name = FAMILIES[family_id]
            confidence = conf.item()
            
            # Generate saliency map
            saliency = DISCOVERY.generate_saliency_map(y_interp, target_class=family_id)
            top_peak_idx = np.argmax(saliency)
            top_peak_wavenumber = common_x[top_peak_idx]
            
            # Create wavenumber array for frontend
            wavenumbers = common_x.tolist()[::10]
            saliency_downsampled = saliency.tolist()[::10]
            spectrum_y_downsampled = y_interp.tolist()[::10]
            
            result = {
                'filename': file.filename,
                'predicted_family': family_name,
                'family_id': family_id,
                'confidence': float(confidence),
                'top_peak_wavenumber': float(top_peak_wavenumber),
                'spectrum': {
                    'x': wavenumbers,
                    'y': spectrum_y_downsampled,
                    'saliency': saliency_downsampled
                },
                'explanation': {
                    'what_is_saliency': 'Shows which parts of your spectrum the AI considered most important for its decision',
                    'confidence_interpretation': f'{confidence:.1%} confident - higher means more reliable',
                    'family_meaning': FAMILIES[family_id] + ' - ' + {
                        0: 'Simple aromatic carbon structures',
                        1: 'Contains nitrogen - potential DNA/RNA building blocks', 
                        2: 'Contains oxygen - may show oxidation or complex chemistry'
                    }.get(family_id, 'Complex organic chemistry')
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
    ensure_data_loaded()
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    matches = RESULTS_DF[RESULTS_DF['id'].str.lower().str.contains(query)]
    results = matches[['id', 'source', 'cluster_membership']].head(50).to_dict('records')
    
    return jsonify(results)

# Load resources immediately for Gunicorn - REMOVED for Lazy Loading pattern
# load_model_and_data()

def ensure_data_loaded():
    """Ensure results data is loaded (lazy loading)"""
    load_results_lazy()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Meteorite Organics Web Application")
    print("="*60)
    print("\n📊 Access the dashboard at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
