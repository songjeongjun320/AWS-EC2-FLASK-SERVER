import os
import pdfplumber
import tempfile
import requests
import logging
import pandas as pd
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

application = Flask(__name__)

# CORS 설정 개선
CORS(application, resources={r"/*": {"origins": [
    "https://business-contract-analyzer.vercel.app",
    "https://business-contract-analyzer-git-main-jun-songs-projects.vercel.app",
    "https://business-contract-analyzer-4252whg8d-jun-songs-projects.vercel.app"
], "methods": ["GET", "POST", "OPTIONS"]}})

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='--Log: %(message)s')

# 파일 크기 제한 설정 (16MB)
application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def process_toxicity(file_path):
    try:
        df = pd.read_excel(file_path)
        df['Calculated Toxicity'] = df['Financial Impact'] * df['Probability of happening']
        
        def categorize_toxicity(toxicity):
            if toxicity <= 25:
                return 'low'
            elif 26 <= toxicity <= 75:
                return 'medium'
            else:
                return 'high'
        
        df['Toxicity Level'] = df['Calculated Toxicity'].apply(categorize_toxicity)
        
        result = {
            'all_items': df['Contractual Terms'].tolist(),
            'high_toxicity_items': df[df['Toxicity Level'] == 'high']['Contractual Terms'].tolist(),
            'medium_toxicity_items': df[df['Toxicity Level'] == 'medium']['Contractual Terms'].tolist(),
            'low_toxicity_items': df[df['Toxicity Level'] == 'low']['Contractual Terms'].tolist()
        }
        
        return json.dumps(result)
    except Exception as e:
        logging.error(f"Error processing Excel file: {str(e)}")
        return json.dumps({'error': str(e)})

@application.route('/')
def home():
    return "<h1>Welcome to the Business Contract Analyzer!</h1>"

@application.route('/process', methods=['POST', 'OPTIONS'])
def process():
    if request.method == 'OPTIONS':
        return '', 204
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.pdf'):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, secure_filename(file.filename))
            file.save(file_path)
            
            txt_files = []
            try:
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        txt_file_path = os.path.join(temp_dir, f"{i + 1}.txt")
                        with open(txt_file_path, 'w') as txt_file:
                            txt_file.write(text or "")
                        txt_files.append(txt_file_path)
                
                frontend_url = request.headers.get('Origin')
                if not frontend_url:
                    return jsonify({"error": "Invalid request: No Origin header"}), 400
                
                upload_endpoint = f"{frontend_url}/api/upload"
                files = {f"file_{i + 1}": open(txt_file, 'rb') for i, txt_file in enumerate(txt_files)}
                
                try:
                    response = requests.post(upload_endpoint, files=files)
                    response.raise_for_status()
                    return jsonify({"message": "Files sent successfully to frontend"}), 200
                except requests.exceptions.RequestException as e:
                    return jsonify({"error": f"Failed to send files to frontend: {str(e)}"}), 500
                finally:
                    for f in files.values():
                        f.close()
            except Exception as e:
                return jsonify({"error": f"Error processing PDF: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400

@application.route('/model_weight', methods=['POST', 'OPTIONS'])
def model_weight():
    if request.method == 'OPTIONS':
        return '', 204
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and (file.filename.endswith('.xlsx') or file.filename.endswith('.xls')):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
            file_path = temp_file.name
            file.save(file_path)
        
        try:
            result_json = process_toxicity(file_path)
            return jsonify(json.loads(result_json)), 200
        except Exception as e:
            return jsonify({"error": f"Failed to process file: {str(e)}"}), 500
        finally:
            os.remove(file_path)
    else:
        return jsonify({"error": "Invalid file type. Please upload an Excel file."}), 400

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)