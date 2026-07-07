import numpy as np
from PIL import Image

def get_prediction(model, image_file):
    """
    Preprocesses the uploaded image and returns the prediction verdict and confidence.
    
    Args:
        model: Loaded Keras model instance.
        image_file: Uploaded image file buffer from Streamlit.
        
    Returns:
        tuple: (label, confidence_score)
               label is either 'Real' or 'Fake'
               confidence_score is a float between 0.0 and 1.0
    """
    try:
        # Validate payload signature if it's a file buffer to prevent arbitrary code execution
        if hasattr(image_file, "read"):
            image_file.seek(0)  # reset position before reading
            header = image_file.read(8)
            image_file.seek(0)  # reset position after reading
            is_jpeg = header.startswith(b'\xff\xd8\xff') or header.startswith(b'\xff\xd8')
            is_png = header.startswith(b'\x89PNG\r\n\x1a\n')
            if not (is_jpeg or is_png):
                raise ValueError("Uploaded payload does not match a valid JPEG or PNG file header signature.")

        # Load and convert to RGB (prevent alpha-channel/grayscale crashes)
        open_image = Image.open(image_file).convert("RGB")
        
        # Resize to expected shape (256, 256)
        resized_image = open_image.resize((256, 256))
        
        # Format as input tensor
        np_image = np.array(resized_image)
        reshaped = np.expand_dims(np_image, axis=0)

        # Run inference
        predicted_prob = model.predict(reshaped)[0][0]
        
        if predicted_prob >= 0.5:
            return "Real", float(predicted_prob)
        else:
            return "Fake", float(1.0 - predicted_prob)
            
    except Exception as e:
        raise RuntimeError(f"Error during preprocessing/inference: {str(e)}")
