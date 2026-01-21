import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Meteorite Organic Analyzer", layout="wide")

st.title("☄️ Meteorite Organics: Spectral Analysis Pipeline")
st.markdown("### Hybrid Supervised–Unsupervised Deep Learning Framework")

st.sidebar.header("Upload Spectrum")
uploaded_file = st.sidebar.file_uploader("Upload a spectral file (.tab, .csv, .txt)", type=["tab", "csv", "txt"])

API_URL = "http://localhost:8000/predict"

if uploaded_file is not None:
    st.sidebar.success("File uploaded!")
    
    # 1. Send to Backend
    if st.sidebar.button("Analyze Spectrum"):
        with st.spinner("Analyzing Chemical Patterns..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                response = requests.post(API_URL, files=files)
                result = response.json()
                
                if "error" in result:
                    st.error(f"Analysis failed: {result['error']}")
                else:
                    # Layout: 2 columns
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("Spectral Analysis & Explainability")
                        # Plot spectrum and saliency map
                        saliency = np.array(result['saliency_map'])
                        # Here we need a common x-axis, assuming 2048 points from 500-4000
                        x_axis = np.linspace(500, 4000, 2048)
                        
                        fig = go.Figure()
                        # Add saliency as background heatmap-like line
                        fig.add_trace(go.Scatter(x=x_axis, y=saliency, name="Saliency (Peakin Contribution)", 
                                                 fill='tozeroy', marker_color='rgba(255, 165, 0, 0.4)'))
                        
                        fig.update_layout(title="Peak Attribution (Saliency Map)", 
                                          xaxis_title="Wavenumber (cm-1)", yaxis_title="Contribution")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info(f"**Top Contributing Peaks:** {', '.join([f'{p:.1f} cm⁻¹' for p in result['top_peaks_cm1']])}")
                    
                    with col2:
                        st.subheader("Chemical Prediction")
                        st.metric("Predicted Family", result['prediction'])
                        st.write(f"**Confidence Score:** {result['confidence']:.2%}")
                        st.progress(result['confidence'])
                        
                        st.subheader("Latent Embedding Location")
                        emb = np.array(result['embedding'])
                        # Dimensionality reduction (PCA/t-SNE) would be better, 
                        # but for now let's just plot two dimensions of the embedding
                        df_emb = pd.DataFrame({'Dim 1': [emb[0]], 'Dim 2': [emb[1]]})
                        st.scatter_chart(df_emb, x='Dim 1', y='Dim 2')
                        
                        st.download_button("Download Analysis Report (JSON)", 
                                           data=str(result), 
                                           file_name=f"report_{result['id']}.json")
            except Exception as e:
                st.error(f"Could not connect to API: {e}. Please ensure the FastAPI backend is running.")

else:
    st.info("Please upload a spectral file to begin the analysis.")
    
    st.markdown("""
    #### Supported Families
    - **Pure PAH:** Polycyclic Aromatic Hydrocarbons (C, H only)
    - **N-PAH:** Nitrogen-containing PAH
    - **O-PAH:** Oxygen-containing PAH
    - **Complex Organic:** Hetero-atomic or complex prebiotic structures
    """)

with st.expander("About the Science"):
    st.write("""
    This framework utilizes a **ResNet-1D architecture** with **Squeeze-and-Excitation blocks** trained on the NASA Ames PAHdb. 
    It enables automated identification and chemical family mapping of organics in extraterrestrial IR spectra, contributing to our understanding of prebiotic chemistry in the early solar system.
    """)
