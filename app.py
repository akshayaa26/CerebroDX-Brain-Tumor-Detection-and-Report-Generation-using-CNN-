import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
from PIL import Image
from fpdf import FPDF

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['REPORT_FOLDER'] = 'reports'

# Load trained model
MODEL_PATH = os.path.join('src', 'final_brain_tumor_model.h5')
model = load_model(MODEL_PATH)
IMG_SIZE = 150
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Tumor details
TUMOR_DETAILS = {
    "glioma_tumor": {
        "description": "Gliomas are tumors that originate in the glial cells of the brain.",
        "risks": "Neurological damage, seizures, loss of motor function.",
        "precautions": "Regular MRI monitoring, symptom awareness.",
        "treatment": "Surgery, radiation, chemotherapy.",
        "procedure": "Biopsy, imaging tests, possible surgical removal.",
        "diet": "High-protein, low-sugar, anti-inflammatory diet.",
        "recommendation": "Consult a neurologist, schedule surgical assessment."
    },
    "meningioma_tumor": {
        "description": "Meningiomas arise from the meninges surrounding the brain.",
        "risks": "Increased intracranial pressure, vision loss.",
        "precautions": "Avoid stress, regular imaging.",
        "treatment": "Surgical removal, radiation.",
        "procedure": "MRI scans, observation or surgery.",
        "diet": "Balanced diet with antioxidants.",
        "recommendation": "Frequent follow-ups and lifestyle adjustments."
    },
    "pituitary_tumor": {
        "description": "Tumors developing on the pituitary gland affecting hormone regulation.",
        "risks": "Hormonal imbalance, vision issues.",
        "precautions": "Regular endocrine evaluations.",
        "treatment": "Medication, surgery, radiation.",
        "procedure": "Hormone tests, imaging, surgical options.",
        "diet": "Hormone-supportive diet (zinc, magnesium).",
        "recommendation": "Endocrinologist referral and treatment planning."
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    symptoms = request.form.getlist('symptoms')
    other_symptoms = request.form['other_symptoms']
    img_file = request.files['image']

    if img_file:
        filename = str(uuid.uuid4()) + "_" + img_file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img_file.save(image_path)

        img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = predictions[predicted_index] * 100

        return render_template('result.html',
                               name=name,
                               age=age,
                               gender=gender,
                               symptoms=', '.join(symptoms),
                               other_symptoms=other_symptoms,
                               prediction=predicted_label,
                               confidence=confidence,
                               image_filename=filename)
    return redirect('/')

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.form
    name = data['name']
    age = data['age']
    gender = data['gender']
    symptoms = data['symptoms']
    other_symptoms = data['other_symptoms']
    prediction = data['prediction']
    confidence = float(data['confidence'])
    image_filename = data['image_filename']
    tumor_info = TUMOR_DETAILS.get(prediction, {})

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(11, 61, 145)
    pdf.cell(0, 10, "CerebroDx - Brain Tumor Detection and Report Generator", ln=True, align='C')

    # Date
    pdf.set_font("Arial", 'I', 11)
    now = datetime.now()
    pdf.cell(0, 10, f"Report Generated on: {now.strftime('%A, %d %B %Y')}", ln=True, align='C')
    pdf.ln(10)

    # Patient Info
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Name: {name}", ln=True)
    pdf.cell(0, 10, f"Age: {age}", ln=True)
    pdf.cell(0, 10, f"Gender: {gender}", ln=True)
    pdf.multi_cell(0, 10, f"Symptoms: {symptoms}")
    if other_symptoms.strip():
        pdf.multi_cell(0, 10, f"Other Symptoms: {other_symptoms}")
    pdf.ln(5)

    # Classification Result
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Tumor Classification", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Predicted Type: {prediction}", ln=True)
    pdf.cell(0, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(5)

    # Tumor Info (only if not 'no_tumor')
    if prediction != "no_tumor" and tumor_info:
        for key, title in [
            ("description", "Tumor Description"),
            ("risks", "Risks"),
            ("precautions", "Precautions"),
            ("treatment", "Treatment Options"),
            ("procedure", "Further Procedures"),
            ("diet", "Recommended Diet"),
            ("recommendation", "Doctor's Recommendation")
        ]:
            pdf.set_font("Arial", 'B', 13)
            pdf.cell(0, 10, f"{title}:", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, tumor_info.get(key, "N/A"))
            pdf.ln(2)

    # MRI Image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    if os.path.exists(image_path):
        try:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "MRI Scan", ln=True)
            pdf.image(image_path, x=40, w=130)
        except:
            pass

    # Save and send PDF
    report_name = f"{name.replace(' ', '_')}_report.pdf"
    report_path = os.path.join(app.config['REPORT_FOLDER'], report_name)
    pdf.output(report_path)

    return send_from_directory(app.config['REPORT_FOLDER'], report_name, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)
    app.run(debug=True)
