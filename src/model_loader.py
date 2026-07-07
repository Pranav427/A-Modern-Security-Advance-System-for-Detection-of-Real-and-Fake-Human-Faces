import os
import streamlit as st
import tensorflow as tf

@st.cache_resource()
def load_detection_model():
    """
    Loads and caches the pretrained face detection model (dffnetv2B0).
    Uses a robust, version-compatible code-based initialization to load weights.
    """
    model_weights_path = "dffnetv2B0.h5"

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(
            f"Model weights file '{model_weights_path}' not found.\n\n"
            "If deploying, please make sure you upload the 'dffnetv2B0.h5' file to your deployment environment."
        )

    try:
        # Dynamically instantiate EfficientNetV2B0 base
        base_model = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights=None,
            input_shape=(256, 256, 3),
            pooling='avg'
        )
        
        # Rename base model to match the weights' group name
        base_model._name = 'efficientnetv2-b0'

        # Wrap in Sequential with custom Dense classification head
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Dense(1, activation='sigmoid', name='dense_2')
        ])

        # Load weights
        model.load_weights(model_weights_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize the TensorFlow/Keras model: {str(e)}")
