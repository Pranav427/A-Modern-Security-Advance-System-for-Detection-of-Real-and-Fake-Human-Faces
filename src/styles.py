import streamlit as st

def apply_custom_styles():
    st.markdown("""
        <style>
        /* General background and fonts */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        /* Title styling */
        .title-text {
            font-weight: 800;
            background: linear-gradient(135deg, #FF4B4B, #FF8383);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-size: 2.8rem;
            margin-bottom: 0.5rem;
        }
        
        .subtitle-text {
            color: #6C7A89;
            text-align: center;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        /* Banner styling */
        .springer-banner {
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            color: #f8fafc;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }
        
        .springer-tag {
            background-color: #f59e0b;
            color: #0f172a;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-block;
            margin-bottom: 0.8rem;
        }

        .springer-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .springer-authors {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-bottom: 0.8rem;
        }

        /* Glassmorphic cards */
        .metric-card {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #FF4B4B;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #8A9Aad;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Prediction results styling */
        .result-container {
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            margin-top: 1rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            border: 1px solid;
        }
        
        .result-real {
            background-color: rgba(16, 185, 129, 0.1);
            border-color: rgba(16, 185, 129, 0.3);
            color: #10b981;
        }

        .result-fake {
            background-color: rgba(239, 68, 68, 0.1);
            border-color: rgba(239, 68, 68, 0.3);
            color: #ef4444;
        }

        .result-verdict {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }

        .result-confidence {
            font-size: 1.2rem;
            font-weight: 500;
            opacity: 0.9;
        }
        
        /* Citation box */
        .citation-box {
            background-color: #0f172a;
            border-radius: 8px;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.8rem;
            overflow-x: auto;
            color: #e2e8f0;
            border: 1px solid #1e293b;
        }

        </style>
    """, unsafe_allow_html=True)
