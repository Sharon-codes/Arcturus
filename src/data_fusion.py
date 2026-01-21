import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from src.data_ingestion import SpectrumObject
from src.preprocessing import SpectralPreprocessor

class DataFusion:
    def __init__(self, x_start=500, x_end=4000, n_points=2048):
        self.common_x = np.linspace(x_start, x_end, n_points)
        self.preprocessor = SpectralPreprocessor()

    def broaden_peaks(self, x_peaks, y_intensities, gamma=15.0):
        """
        Converts discrete peaks into a continuous spectrum using Lorentzian broadening.
        gamma: Half-width at half-maximum (HWHM).
        """
        spectrum = np.zeros_like(self.common_x)
        for xp, yi in zip(x_peaks, y_intensities):
            # Lorentzian profile
            spectrum += yi * (gamma**2 / ((self.common_x - xp)**2 + gamma**2))
        return spectrum

    def process_and_merge(self, spectra: List[SpectrumObject]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Transforms all SpectrumObjects into a unified matrix aligned to common_x.
        Returns:
            X: (N, n_points) matrix of spectra
            y: (N,) vector of labels (if available)
            metadata: pd.DataFrame of metadata
        """
        X_list = []
        labels = []
        meta_list = []

        print(f"Aligning and processing {len(spectra)} spectra...", flush=True)
        for s in tqdm(spectra, desc="Processing Spectra", unit="spec"):
            if s.source == "PAHdb":
                # Theoretical peaks -> Broadened continuous spectrum
                y_cont = self.broaden_peaks(s.x, s.y)
                # Apply same preprocessing as real data (except baseline correction which isn't needed)
                y_proc = self.preprocessor.normalize(y_cont, method='max')
            else:
                # Experimental PDS -> Preprocess and interpolate
                _, y_proc = self.preprocessor.full_pipeline(s.x, s.y, target_x=self.common_x)
            
            X_list.append(y_proc)
            labels.append(s.label if s.label is not None else 0)
            meta_list.append({
                "id": s.id,
                "source": s.source,
                **s.metadata
            })

        X = np.stack(X_list)
        y = np.array(labels)
        metadata = pd.DataFrame(meta_list)

        return X, y, metadata

if __name__ == "__main__":
    fusion = DataFusion()
    print(f"Unified x-axis created with {len(fusion.common_x)} points.")
