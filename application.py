import os
import pdfplumber
import tempfile
import logging
import pandas as pd
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import time
from groq import Groq  # Assuming you have a Groq Python SDK; install if needed
from dotenv import load_dotenv

application = Flask(__name__)

# CORS 설정 추가 (모든 도메인 허용)
CORS(application, resources={r"/*": {"origins": "*"}})

# Logging 설정
logging.basicConfig(level=logging.INFO, format='--Log: %(message)s')
load_dotenv()  # Load environment variables

# groq API 호출
async def process_groq(txt_dir, json_dir):
    logging.info("process_groq function started")

    # Paths to JSON files
    base_data_path = os.path.join(json_dir, "base_data.json")
    all_results_path = os.path.join(json_dir, "all_results.json")

    # Load toxicity level data from base_data.json
    try:
        with open(base_data_path, 'r', encoding="utf-8") as f:
            base_data = json.load(f)
        with open(all_results_path, 'r', encoding="utf-8") as f:
            all_result_data = json.load(f)

        logging.info("Loaded base_data.json and all_results.json successfully")
    except Exception as error:
        logging.error(f"Error loading base_data.json: {error}")
        raise error

    # Read text files
    text_files = [file for file in os.listdir(txt_dir) if file.endswith(".txt")]
    logging.info(f"Text files to process: {text_files}")

    # Initialize Groq SDK
    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Process each text file without any forced delay
    for file_name in text_files:
        file_path = os.path.join(txt_dir, file_name)
        logging.info(f"Starting processing for file: {file_name}")

        # Try reading the file with UTF-8, fallback to ISO-8859-1 if UTF-8 fails
        try:
            with open(file_path, 'r', encoding="utf-8") as file:
                text = file.read()
        except UnicodeDecodeError:
            logging.warning(f"UTF-8 decoding failed for {file_name}. Trying ISO-8859-1.")
            with open(file_path, 'r', encoding="ISO-8859-1") as file:
                text = file.read()

        start_time = time.time()
        logging.info(f"Sending request to Groq API for file: {file_name}")

        try:
            # Send request to Groq API
            response = await groq.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"This is a base_data.json file containing keys that represent clauses in a business contract: {json.dumps(all_result_data)}. Analyze the provided text and categorize the clauses according to these keys. Follow the rules below:\n"
                                   "1. Do not create new keys.\n"
                                   "2. Only use the existing keys from base_data.json.\n"
                                   "3. Respond with a JSON format that matches the exact structure of base_data.json.\n"
                                   "4. Extract relevant sentences from the provided text and add them as values in string format under the appropriate key in base_data.json.\n"
                                   "5. If the relevant sentence is too long, summarize it to 1 or 2 sentences."
                    },
                    {
                        "role": "user",
                        "content": f"Ensure the response format matches base_data.json. No comments, Just .json format\n\n{text}"
                    }
                ],
                model="llama3-70b-8192"
            )

            # Process the response and save results
            json_content = response.choices[0].message.content.strip()
            if '{' in json_content and '}' in json_content:
                json_content = json_content[json_content.index('{'):json_content.rindex('}') + 1]

            categorized_clauses = json.loads(json_content)
            result_file_path = os.path.join(json_dir, "all_results.json")

            # Merge results with existing data
            try:
                with open(result_file_path, 'r', encoding="utf-8") as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                existing_data = {}

            for key, value in categorized_clauses.items():
                if isinstance(existing_data.get(key), list):
                    existing_data[key].extend(value)
                else:
                    existing_data[key] = value

            with open(result_file_path, 'w', encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2)

            elapsed_time = time.time() - start_time
            logging.info(f"File processed and saved: {file_name} (Time taken: {elapsed_time:.2f}s)")
        
        except Exception as error:
            logging.error(f"Error processing file {file_name}: {error}")

@application.route('/process', methods=['POST'])
def process():
    logging.info("Processing file upload")

    # 파일 유무 체크
    if 'file' not in request.files:
        logging.error("No file part in request")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected")
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        logging.info(f"Received PDF file: {file.filename}")
        
        # PDF 파일을 임시 디렉토리에 저장
        temp_dir = tempfile.gettempdir()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=temp_dir) as temp_file:
            file_path = temp_file.name
            file.save(file_path)
        logging.info(f"Temporary PDF file saved at: {file_path}")

        # PDF를 페이지별로 텍스트 파일로 변환
        txt_dir = os.path.join(temp_dir, "txt_results")
        os.makedirs(txt_dir, exist_ok=True)
        txt_files = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    txt_file_path = os.path.join(txt_dir, f"{i + 1}.txt")
                    with open(txt_file_path, 'w') as txt_file:
                        txt_file.write(text or "")
                    txt_files.append(txt_file_path)
                    logging.info(f"Created text file for page {i + 1}: {txt_file_path}")

            # JSON 결과를 저장할 디렉토리 생성
            json_dir = os.path.join(temp_dir, "json_results")
            os.makedirs(json_dir, exist_ok=True)
            logging.info(f"Created JSON result directory: {json_dir}")

            # 비동기 함수 호출
            asyncio.run(process_groq(txt_dir, json_dir))

            # JSON 응답으로 반환
            final_results_path = os.path.join(json_dir, "final_results.json")
            with open(final_results_path, 'r', encoding="utf-8") as f:
                final_results_data = json.load(f)
            return jsonify(final_results_data), 200

        finally:
            # 임시 PDF 파일 삭제
            os.remove(file_path)
            for txt_file in txt_files:
                try:
                    os.remove(txt_file)
                    logging.info(f"Deleted {txt_file}")
                except Exception as e:
                    logging.error(f"Failed to delete {txt_file}: {e}")
    else:
        logging.error("Invalid file type")
        return jsonify({"error": "Invalid file type"}), 400
    
    
@application.route('/')
def health_check():
    return jsonify({"status": "running"}), 200

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)
