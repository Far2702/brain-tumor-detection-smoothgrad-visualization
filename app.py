from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO
import os

# Load the trained model
model=load_model('model_final.h5')

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file=request.files['file']
    image=Image.open(file).convert("RGB").resize((224, 224))
    img_array=np.expand_dims(np.array(image)/255.0, axis=0)
    prediction=model.predict(img_array)[0][0]
    if prediction>0.5:
        label="Tumor"
        confidence=prediction*100
    else:
        label="No Tumor Present"
        confidence=1-prediction
    

    file.seek(0)
    file.save('static/latest_upload.png')  
    return render_template("result.html", prediction=label, confidence=confidence)

@app.route('/smoothgrad')
def smoothgrad():
    image_path='static/latest_upload.png'
    
    if not os.path.exists(image_path):
        return "No image found. Please upload and predict first."
    
    
    # Load the saved input image
    image=Image.open("static/latest_upload.png").convert("RGB").resize((224, 224))
    img_array=np.expand_dims(np.array(image)/255.0,axis=0)

    model_input=tf.convert_to_tensor(img_array,dtype=tf.float32)
    smooth_grads=[]
    num_samples=50
    noise_level=0.10
    class_index=0
    for i in range(num_samples):
        noise=tf.random.normal(shape=model_input.shape,stddev=noise_level)
        noisy_img=model_input+noise
        with tf.GradientTape() as tape:
            tape.watch(noisy_img)
            preds=model(noisy_img,training=False)
            loss=preds[:,class_index]
        grads=tape.gradient(loss, noisy_img)
        saliency=tf.reduce_max(tf.abs(grads),axis=-1)[0]
        smooth_grads.append(saliency.numpy())

    avg_grads=np.mean(np.stack(smooth_grads), axis=0)
    avg_grads= cv2.GaussianBlur(avg_grads, (5, 5), 0.5)
    vmin, vmax=np.percentile(avg_grads, [2, 98])
    avg_grads=np.clip(avg_grads, vmin, vmax)
    avg_grads=(avg_grads-vmin)/(vmax-vmin+1e-8)
    
    # Create heatmap
    original_img=np.uint8(img_array[0]*255)
    heatmap=cv2.applyColorMap(np.uint8(255 * avg_grads), cv2.COLORMAP_JET)
    heatmap=cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
    overlay_img=cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    pil_img=Image.fromarray(overlay_img)
    buffer=BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")


    return render_template("smoothgrad.html", smoothgrad_img=encoded_img)











app.run(debug=True)