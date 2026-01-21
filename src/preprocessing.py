import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Tuple

class SpectralPreprocessor:
    def __init__(self):
        pass

    def baseline_als(self, y, lam=1e5, p=0.01, niter=10):
        """
        Asymmetric Least Squares Smoothing for baseline correction.
        Paper: "Baseline Correction with Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens.
        """
        L = len(y)
        D = sparse.diags([1,-2,1], [0,-1,-2], shape=(L,L-2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            # Convert to CSR for efficient solving
            z = spsolve(Z.tocsr(), w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z

    def smooth(self, y, window_length=11, polyorder=3):
        """Savitzky-Golay smoothing."""
        if len(y) <= window_length:
            return y
        return savgol_filter(y, window_length, polyorder)

    def normalize(self, y, method='max'):
        """Normalize spectrum (max, area, or scaling)."""
        if method == 'max':
            return y / np.max(y) if np.max(y) != 0 else y
        elif method == 'area':
            area = np.trapz(y)
            return y / area if area != 0 else y
        return y

    def interpolate_spectrum(self, x, y, new_x):
        """Align spectrum to a common x-axis."""
        f = interp1d(x, y, kind='linear', fill_value="extrapolate")
        return f(new_x)

    def full_pipeline(self, x, y, target_x=None):
        """Full preprocessing pipeline."""
        # 1. Smoothing
        y_proc = self.smooth(y)
        
        # 2. Baseline correction
        baseline = self.baseline_als(y_proc)
        y_proc = y_proc - baseline
        
        # 3. Handle negatives after baseline
        y_proc = np.maximum(y_proc, 0)
        
        # 4. Normalization
        y_proc = self.normalize(y_proc, method='max')
        
        # 5. Interpolation (if target_x provided)
        if target_x is not None:
            y_proc = self.interpolate_spectrum(x, y_proc, target_x)
            return target_x, y_proc
            
        return x, y_proc
    
    def preprocess(self, x, y):
        """Alias for full_pipeline for API consistency"""
        return self.full_pipeline(x, y)
    
    def baseline_correction(self, y):
        """Alias for baseline_als for API consistency"""
        return self.baseline_als(y)

if __name__ == "__main__":
    # Test on dummy data
    x = np.linspace(0, 100, 1000)
    y = np.exp(-((x-50)**2)/10) + 0.1*x # peak + sloped baseline
    preprocessor = SpectralPreprocessor()
    x_proc, y_proc = preprocessor.full_pipeline(x, y)
    print("Preprocessing complete (dummy test)")
