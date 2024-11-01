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

# CORS 설정 추가
CORS(application, resources={r"/*": {"origins": "*"}})

# Set up logging
logging.basicConfig(level=logging.INFO, format='--Log: %(message)s')
load_dotenv()  # Load environment variables
logging.basicConfig(level=logging.INFO)

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

        logging.info("Successfully loaded toxicity level data from base_data.json")
    except Exception as error:
        logging.error("Error loading base_data.json:", error)
        raise error

    # Read text files
    text_files = [file for file in os.listdir(txt_dir) if file.endswith(".txt")]
    logging.info(f"Text files found for processing: {text_files}")

    # Initialize Groq SDK
    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # 기본 대기 시간 설정
    base_delay = 20  # 기본 대기 시간
    delay = base_delay  # 초기 대기 시간 설정

    # Process each text file with a delay to prevent rate limiting
    for file_name in text_files:
        file_path = os.path.join(txt_dir, file_name)

        # Try reading the file with UTF-8, fall back to ISO-8859-1 if UTF-8 fails
        try:
            with open(file_path, 'r', encoding="utf-8") as file:
                text = file.read()
            logging.info(f"Successfully read text from file: {file_name}")
        except UnicodeDecodeError:
            logging.warning(f"UTF-8 decoding failed for {file_name}. Trying ISO-8859-1.")
            with open(file_path, 'r', encoding="ISO-8859-1") as file:
                text = file.read()

        logging.info(f"Processing file: {file_name}")

        try:
            # Send request to Groq API
            response = groq.chat.completions.create(
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

            logging.info(f"Sent request to Groq API for file: {file_name}")

            # Process the response and save results
            json_content = response.choices[0].message.content.strip()
            if '{' in json_content and '}' in json_content:
                json_content = json_content[json_content.index('{'):json_content.rindex('}') + 1]

            categorized_clauses = json.loads(json_content)
            result_file_path = os.path.join(json_dir, "all_results.json")

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

            logging.info(f"Updated results saved in all_results.json for {file_name}")

            # 파일 처리가 완료된 후 대기 시간을 기다리지 않고 다음 파일로 이동
            delay = base_delay  # 대기 시간을 기본 대기 시간으로 초기화

        except Exception as error:
            logging.error(f"Error processing file {file_name}: {error}")

        # Add a delay between requests to avoid hitting rate limits
        await asyncio.sleep(delay)  # Adjust the delay (in seconds) as needed



@application.route('/')
def home():
    logging.info("Home route accessed")
    return jsonify({"message": "Welcome to the Business Contract Analyzer!"}), 200


# Getting .pdf and process them
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

            # model_weight 함수를 호출하여 결과를 생성
            result_json = model_weight()
            result = json.loads(result_json.get_data(as_text=True))  # model_weight 함수에서 반환된 JSON 결과를 로드

            # JSON 파일들 생성 및 저장
            all_results_data = {item: [] for item in result['all_items']}
            with open(os.path.join(json_dir, "all_results.json"), 'w', encoding="utf-8") as f:
                json.dump(all_results_data, f, indent=2)
            logging.info("Created all_results.json")

            base_data = {
                "high": result['high_toxicity_items'] or [],
                "medium": result['medium_toxicity_items'] or [],
                "low": result['low_toxicity_items'] or []
            }
            with open(os.path.join(json_dir, "base_data.json"), 'w', encoding="utf-8") as f:
                json.dump(base_data, f, indent=2)
            logging.info("Created base_data.json")

            final_results_data = {"high": [], "medium": [], "low": []}
            with open(os.path.join(json_dir, "final_results.json"), 'w', encoding="utf-8") as f:
                json.dump(final_results_data, f, indent=2)
            logging.info("Created final_results.json")

            # process_groq 및 organize_final_results 비동기 함수 호출
            asyncio.run(process_groq(txt_dir, json_dir))
            asyncio.run(organize_final_results(json_dir))

            # final_results.json 파일 경로 설정
            final_results_path = os.path.join(json_dir, "final_results.json")

            # 클라이언트에 final_results.json을 JSON 응답으로 반환
            with open(final_results_path, 'r', encoding="utf-8") as f:
                final_results_data = json.load(f)

            # final_results.json 그대로 전달
            logging.info("Returning final results to client")
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
    

# 루트 디렉토리의 weights.xlsx 파일을 사용하여 모델 가중치 계산
def model_weight():
    logging.info("루트 디렉토리의 weights.xlsx 파일로부터 독성 모델을 처리합니다")

    # 루트 디렉토리의 weights.xlsx 파일 경로 설정
    file_path = os.path.join(os.getcwd(), "weights.xlsx")
    
    # weights.xlsx 파일이 실제로 존재하는지 확인
    if not os.path.exists(file_path):
        logging.error("루트 디렉토리에 weights.xlsx 파일이 존재하지 않습니다")
        return jsonify({"error": "weights.xlsx 파일을 찾을 수 없습니다"})

    try:
        # 엑셀 파일을 처리
        result_json = process_toxicity(file_path)
        logging.info("엑셀 파일이 성공적으로 처리되었습니다")

        # 처리 결과를 JSON 형식으로 반환
        return jsonify(json.loads(result_json))

    except Exception as e:
        logging.error(f"엑셀 파일 처리 중 오류 발생: {e}")
        return jsonify({"error": f"파일 처리에 실패했습니다: {str(e)}"})


# 기존의 process_toxicity 함수 추가
def process_toxicity(file_path):
    """
    주어진 엑셀 파일을 읽고, 독성 수준을 계산하여 분류한 후,
    결과를 JSON 형식으로 반환합니다.
    """
    try:
        # 엑셀 파일 읽기
        df = pd.read_excel(file_path)
        logging.info(f"Reading Excel file: {file_path}")

        # 'Financial Impact'와 'Probability of happening'을 곱해서 새로운 'Calculated Toxicity' 열 추가
        df['Calculated Toxicity'] = df['Financial Impact'] * df['Probability of happening']
        logging.info("Calculated toxicity levels based on financial impact and probability.")

        # 독성 수준을 분류하는 함수
        def categorize_toxicity(toxicity):
            if toxicity <= 25:
                return 'low'
            elif 26 <= toxicity <= 75:
                return 'medium'
            else:
                return 'high'

        # 분류 함수 적용해서 'Toxicity Level' 열 추가
        df['Toxicity Level'] = df['Calculated Toxicity'].apply(categorize_toxicity)
        logging.info("Categorized toxicity levels into low, medium, and high.")

        # 전체 항목 리스트
        all_items_list = df['Contractual Terms'].tolist()
        logging.info("Extracted all contractual terms.")

        # 'high', 'medium', 'low'로 그룹화된 리스트 생성
        high_list = df[df['Toxicity Level'] == 'high']['Contractual Terms'].tolist()
        medium_list = df[df['Toxicity Level'] == 'medium']['Contractual Terms'].tolist()
        low_list = df[df['Toxicity Level'] == 'low']['Contractual Terms'].tolist()

        result = {
            'all_items': all_items_list,
            'high_toxicity_items': high_list,
            'medium_toxicity_items': medium_list,
            'low_toxicity_items': low_list
        }

        # JSON 형식으로 결과를 반환
        logging.info("Processed toxicity data and preparing JSON response.")
        return json.dumps(result)

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return json.dumps({'error': f"File not found: {file_path}"})
    except pd.errors.EmptyDataError:
        logging.error("Excel file is empty.")
        return json.dumps({'error': "Excel file is empty"})
    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        return json.dumps({'error': f"Missing expected column: {e}"})
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        return json.dumps({'error': f"An unexpected error occurred: {str(e)}"})


# process_groq 실행 후 결과 정리
async def organize_final_results(json_dir):
    # 파일 경로 설정
    all_results_path = os.path.join(json_dir, "all_results.json")
    base_data_path = os.path.join(json_dir, "base_data.json")
    final_results_path = os.path.join(json_dir, "final_results.json")

    # 파일 읽기
    try:
        with open(all_results_path, 'r', encoding="utf-8") as f:
            all_results_data = json.load(f)
        with open(base_data_path, 'r', encoding="utf-8") as f:
            base_data = json.load(f)
        
        # 초기화된 final_results 데이터 구조
        final_results_data = {"high": [], "medium": [], "low": []}
        logging.info("Initialized final results data structure.")

        # base_data를 기준으로 all_results 데이터를 분류하여 final_results에 추가
        for item, clauses in all_results_data.items():
            if item in base_data['high']:
                final_results_data['high'].extend(clauses)
                logging.info(f"Added clauses to high category for item: {item}")
            elif item in base_data['medium']:
                final_results_data['medium'].extend(clauses)
                logging.info(f"Added clauses to medium category for item: {item}")
            elif item in base_data['low']:
                final_results_data['low'].extend(clauses)
                logging.info(f"Added clauses to low category for item: {item}")

        # 결과를 final_results.json 파일로 저장
        with open(final_results_path, 'w', encoding="utf-8") as f:
            json.dump(final_results_data, f, indent=2)
        logging.info("Organized data saved in final_results.json")

    except Exception as e:
        logging.error(f"Error organizing final results: {e}")
        raise


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)
