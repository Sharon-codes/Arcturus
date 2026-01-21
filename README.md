# ARCTURUS: Meteorite Organic Discovery ☄️

**Arcturus** is a hybrid supervised–unsupervised deep learning framework designed to automate the identification and chemical family mapping of prebiotic organics in extraterrestrial infrared (IR) spectra. This research automates the search for the "seeds of life" in meteorite samples.

## 🔬 Scientific Framework

The core of Arcturus is a Deep Neural Network trained on the **NASA Ames PAHdb** (Polycyclic Aromatic Hydrocarbon Database). It leverages a **ResNet-1D architecture** with Squeeze-and-Excitation (SE) blocks to extract meaningful features from complex, noisy spectral data.

### Key Capabilities
- **Automated Classification**: Instantly categorizes spectra into structural families:
  - **Pure PAHs**: The inert carbon backbone.
  - **N-PAHs / O-PAHs**: Nitrogen and Oxygen-substituted variants (Prebiotic precursors).
  - **Aliphatics**: Primitive, chain-like hydrocarbons.
- **Saliency Mapping**: Uses "AI Saliency" to highlight exactly *where* in the spectrum the neural network is looking, providing interpretability for the classification.
- **Hybrid Embeddings**: Maps spectra into a latent space to find clusters of chemistries, bridging the gap between known laboratory samples and unknown astronomical observations.

## 🚀 Application Stack

This repository contains the web application interface for Arcturus.

- **Backend**: Flask (Python) serving the PyTorch model inference.
- **Frontend**: HTML5/CSS3 with a "Space/Sci-Fi" aesthetic, featuring dynamic JS plotting (Plotly.js) and responsive design.
- **Model**: PyTorch ResNet-1D (`best_model.pth`).

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sharon-codes/Arcturus.git
   cd Arcturus
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   python webapp.py
   ```
   The dashboard will be available at `http://localhost:5000`.

## 📊 Research Context

**Presented at:** Indian Institute of Space Science and Technology (IIST)
**Author:** Sharon Melhi, Amity University Bengaluru

*> "We scan meteorite spectra to find the chemical building blocks of existence."*
