from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import torch
from openai import OpenAI

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# OpenAI API configuration
OPENAI_API_KEY = "api key"
client = OpenAI(api_key=OPENAI_API_KEY)

# Load the fine-tuned BLIP model and processor
blip_model = BlipForConditionalGeneration.from_pretrained("flask_backend/textmodel").to(device)
blip_processor = BlipProcessor.from_pretrained("flask_backend/textmodel")

# Load the pre-trained VGG model
vgg_model = load_model("flask_backend/pneumonia_detection_vgg_model.h5")

# Define class labels for the VGG model
class_labels = ['NORMAL', 'bacteria', 'virus']

# Helper function: Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper function: Process image for VGG model
def process_image_for_vgg(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Helper function: Generate BLIP report
def generate_report_with_blip(file_path):
    try:
        img = Image.open(file_path).convert("RGB")
        inputs = blip_processor(images=img, return_tensors="pt").to(device)
        blip_model.eval()
        with torch.no_grad():
            generated_ids = blip_model.generate(pixel_values=inputs["pixel_values"])
            return blip_processor.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error in BLIP report generation: {e}")
        return "BLIP report generation failed."

# Helper function: Generate detailed medical report
def generate_medical_report(diagnosis, confidence, explanation, blip_report):
    prompt = (
        "Generate a detailed medical report including technical causes of the diagnosis. "
        "Divide the report into three sections:\n"
        "1. Biological Causes: Detailed biological reasons leading to the diagnosis.\n"
        "2. Pathological Analysis: Insights from pathological observations or studies.\n"
        "3. Environmental Factors: Any relevant environmental or lifestyle causes.\n"
        "Explain each section in detail as if teaching a medical student, without patient-specific details."
        f"\nDiagnosis: {diagnosis}\n"
        f"Confidence: {confidence}\n"
        f"Explanation: {explanation}\n"
        f"BLIP Report: {blip_report}"
    )
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical assistant providing detailed diagnoses."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Split the response into sections based on expected structure
    response_text = completion.choices[0].message.content
    sections = {"Biological Causes": "", "Pathological Analysis": "", "Environmental Factors": ""}
    
    for section in sections:
        start_index = response_text.find(section)
        if start_index != -1:
            end_index = response_text.find("\n", start_index + len(section))
            next_section_index = min(
                [response_text.find(next_key, end_index) for next_key in sections if next_key != section and response_text.find(next_key, end_index) != -1],
                default=None
            )
            sections[section] = response_text[start_index:end_index if next_section_index is None else next_section_index].strip()
    
    return sections


# Endpoint: Analyze image
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # VGG model prediction
        vgg_input = process_image_for_vgg(file_path)
        vgg_prediction = vgg_model.predict(vgg_input)
        confidence = np.max(vgg_prediction)
        predicted_class = class_labels[np.argmax(vgg_prediction)]

        # BLIP model report
        blip_report = generate_report_with_blip(file_path)

        # Generate structured medical report
        structured_report = generate_medical_report(
            diagnosis=predicted_class,
            confidence=f"{confidence:.2%}",
            explanation=f"Detected {predicted_class} pneumonia with {confidence:.2%} confidence.",
            blip_report=blip_report
        )

        return jsonify({
            'structured_report': {
                'biological_causes': structured_report.get("Biological Causes", ""),
                'pathological_analysis': structured_report.get("Pathological Analysis", ""),
                'environmental_factors': structured_report.get("Environmental Factors", "")
            },
            'diagnosis': predicted_class,
            'confidence': f"{confidence:.2%}",
            'blip_report': blip_report
        }), 200
    else:
        return jsonify({'error': 'Invalid file format. Only jpg, jpeg, png allowed.'}), 400

# Endpoint: Chat for follow-up questions
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Invalid input. Provide a question.'}), 400

    question = data['question']
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical expert answering follow-up questions."},
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content
        return jsonify({'answer': answer}), 200
    except Exception as e:
        print(f"Error in chat response: {e}")
        return jsonify({'error': 'Chat response generation failed.'}), 500

# Run Flask app
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
