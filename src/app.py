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
                        st.subheader("🔍 AI Decision Analysis")
                        
                        # Create tabs for different views
                        tab1, tab2 = st.tabs(["📊 Spectrum + AI Focus", "🎯 Key Insights"])
                        
                        with tab1:
                            # Plot spectrum and saliency map
                            saliency = np.array(result['saliency_map'])
                            x_axis = np.linspace(500, 4000, len(saliency))
                            
                            fig = go.Figure()
                            
                            # Add spectrum
                            fig.add_trace(go.Scatter(
                                x=x_axis, y=np.array(result['spectrum_y']),
                                name="Your Spectrum", 
                                line=dict(color='#00f3ff', width=2),
                                fill='tozeroy', 
                                fillcolor='rgba(0, 243, 255, 0.1)'
                            ))
                            
                            # Add saliency as background highlighting
                            max_intensity = np.max(result['spectrum_y'])
                            saliency_scaled = saliency * max_intensity * 0.3
                            fig.add_trace(go.Scatter(
                                x=x_axis, 
                                y=saliency_scaled,
                                name="AI Attention Areas",
                                fill='tozeroy',
                                fillcolor='rgba(255, 0, 60, 0.3)',
                                line=dict(width=0),
                                hovertemplate='Wavenumber: %{x:.1f} cm⁻¹<br>AI Attention: %{customdata:.3f}<extra></extra>',
                                customdata=saliency
                            ))
                            
                            # Add saliency line
                            fig.add_trace(go.Scatter(
                                x=x_axis, y=saliency,
                                name="AI Decision Strength", 
                                yaxis="y2",
                                line=dict(color='#ff003c', width=1.5, dash='dot')
                            ))
                            
                            fig.update_layout(
                                title="Your Spectrum with AI Analysis",
                                xaxis_title="Wavenumber (cm⁻¹) - Higher numbers = smaller molecular vibrations",
                                yaxis_title="Light Intensity",
                                yaxis2=dict(
                                    title="AI Attention Level",
                                    overlaying="y",
                                    side="right",
                                    showgrid=False
                                ),
                                annotations=[dict(
                                    x=0.5, y=1.1, xref="paper", yref="paper",
                                    text="🔍 Pink areas show where the AI looked most closely",
                                    showarrow=False, font=dict(size=12, color='#ff6a00')
                                )]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab2:
                            st.markdown("### 🎯 What the AI Found Important")
                            
                            # Find top peaks
                            top_indices = np.argsort(saliency)[-5:][::-1]  # Top 5 peaks
                            
                            for i, idx in enumerate(top_indices):
                                wn = x_axis[idx]
                                importance = saliency[idx]
                                
                                # Determine region
                                if wn > 3000:
                                    region = "🔬 C-H Stretch (Carbon-Hydrogen bonds)"
                                    meaning = "Shows the basic carbon framework of organic molecules"
                                elif wn > 2000:
                                    region = "🧬 Triple Bonds (C≡N or C≡C)"
                                    meaning = "Indicates nitriles or alkynes - potential prebiotic chemistry"
                                elif wn > 1500:
                                    region = "🌟 Double Bonds & Aromatics"
                                    meaning = "Shows aromatic rings or carbonyl groups"
                                else:
                                    region = "🔍 Complex Fingerprints"
                                    meaning = "Unique patterns from complex molecular structures"
                                
                                st.markdown(f"""
                                **#{i+1} Key Feature** (Attention: {importance:.3f})
                                - **Location**: {wn:.1f} cm⁻¹
                                - **Region**: {region}
                                - **Significance**: {meaning}
                                """)
                            
                            st.info(f"💡 **Overall**: The AI is {result['confidence']:.1%} confident this is **{result['prediction']}** based on these spectral patterns.")
                    
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
    st.info("👆 Please upload a spectral file (.tab, .csv, .txt) to begin the analysis.")
    
    st.markdown("""
    ## 🤔 How Does the AI Analysis Work?
    
    When you upload a spectrum, our AI:
    
    1. **Reads your data** - Infrared light measurements showing how molecules vibrate
    2. **Compares patterns** - Matches against thousands of known organic compounds  
    3. **Shows its reasoning** - Highlights which parts of your spectrum were most important
    4. **Makes a prediction** - Classifies the chemical family with confidence score
    
    ### 📊 Understanding the Results
    
    - **Blue line**: Your original spectrum (light intensity vs. molecular vibration frequency)
    - **Pink areas**: Where the AI paid closest attention - these "hotspots" drove the decision
    - **Red dots**: Key decision points with explanations of what they represent
    - **Confidence score**: How sure the AI is (higher = more reliable)
    
    ### 🔬 Chemical Families
    
    - **Pure PAH**: Simple aromatic hydrocarbons (like graphite-like structures)
    - **N-PAH**: Contains nitrogen - potential building blocks for DNA/RNA
    - **O-PAH**: Contains oxygen - may indicate oxidation or complex organics
    """)

with st.expander("About the Science"):
    st.write("""
    This framework utilizes a **ResNet-1D architecture** with **Squeeze-and-Excitation blocks** trained on the NASA Ames PAHdb. 
    It enables automated identification and chemical family mapping of organics in extraterrestrial IR spectra, contributing to our understanding of prebiotic chemistry in the early solar system.
    """)
