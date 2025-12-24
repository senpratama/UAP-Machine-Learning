import numpy as np

def predict_top5(model, image_array, class_names):
    # Lakukan prediksi
    predictions = model.predict(image_array, verbose=0)[0]
    
    # Ambil index urutan dari yang tertinggi
    top_indices = predictions.argsort()[-5:][::-1]
    
    results = []
    for i in top_indices:
        results.append({
            "class": class_names[i],
            "confidence": float(predictions[i] * 100)
        })
    
    return results