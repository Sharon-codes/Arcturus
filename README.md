# ARCTURUS: Meteorite Organic Discovery ☄️

**Arcturus** is a hybrid supervised–unsupervised deep learning framework designed to automate the identification and chemical family mapping of prebiotic organics in extraterrestrial infrared (IR) spectra. This research automates the search for the "seeds of life" in meteorite samples.

## 🔬 Scientific Framework

The core of Arcturus is a Deep Neural Network trained on the **NASA Ames PAHdb** (Polycyclic Aromatic Hydrocarbon Database). It leverages a **ResNet-1D architecture** with Squeeze-and-Excitation (SE) blocks to extract meaningful features from complex, noisy spectral data.

### Key Capabilities
- **Automated Classification**: Instantly categorizes spectra into structural families:
  - **Pure PAHs**: The inert carbon backbone.
  - **N-PAHs / O-PAHs**: Nitrogen and Oxygen-substituted variants (Prebiotic precursors).
  - **Aliphatics**: Primitive, chain-like hydrocarbons.
- **AI Saliency Maps**: Shows exactly which parts of your spectrum influenced the AI's decision most
- **Hybrid Embeddings**: Maps spectra into a latent space to find clusters of chemistries, bridging the gap between known laboratory samples and unknown astronomical observations.

## 🤔 Understanding AI Saliency Maps

When you upload a spectrum, the AI doesn't just give a yes/no answer - it shows its "thought process" through **saliency maps**:

### What Are Saliency Maps?
- **Visual explanation** of what the AI considered important in your spectrum
- **Pink/red highlighted areas** show where the AI focused most
- **Higher peaks = stronger evidence** used in the classification

### How to Read Them
1. **Blue line**: Your original spectrum (light intensity vs. molecular vibration)
2. **Pink background**: AI attention areas - where the AI looked closely
3. **Red markers**: Key decision points with explanations
4. **Confidence score**: How sure the AI is (85%+ is very reliable)

### What Do Different Regions Mean?
- **3000+ cm⁻¹**: C-H bonds (basic organic structure)
- **2000-2300 cm⁻¹**: Triple bonds (C≡N, C≡C) - potential prebiotic chemistry
- **1500-2000 cm⁻¹**: Double bonds & aromatics (ring structures)
- **<1500 cm⁻¹**: Complex fingerprints (unique molecular signatures)

### Example Interpretation
If the AI highlights peaks around 2200 cm⁻¹ and classifies as "N-PAH" with 92% confidence, it means:
- The AI found strong evidence of nitrile groups (C≡N)
- This suggests nitrogen-containing organic compounds
- High confidence means this pattern matches known laboratory samples well

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
