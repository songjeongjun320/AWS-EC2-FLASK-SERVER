import os
import pdfplumber
import tempfile
import logging
import pandas as pd
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
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
def process_groq(txt_dir, json_dir):
    logging.info("process_groq function started")

    # 전체 작업 시작 시간 기록
    total_start_time = time.time()

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
    groq = Groq(api_key=os.getenv("GROQ_API_KEY_SUB1"))

    # 기본 대기 시간 설정
    base_delay = 10  # 시작 대기 시간
    max_delay = 60   # 최대 대기 시간

    # Process each text file with dynamic delay based on response
    for file_name in text_files:
        file_path = os.path.join(txt_dir, file_name)
        logging.info(f"Starting processing for file: {file_name}")

        # 파일 읽기 시도
        try:
            with open(file_path, 'r', encoding="utf-8") as file:
                text = file.read()
        except UnicodeDecodeError:
            logging.warning(f"UTF-8 decoding failed for {file_name}. Trying ISO-8859-1.")
            with open(file_path, 'r', encoding="ISO-8859-1") as file:
                text = file.read()

        delay = base_delay  # 초기 대기 시간 설정
        start_time = time.time()  # 파일 처리 시작 시간 기록

        while True:
            try:
                logging.info(f"Sending request to Groq API for file: {file_name}")

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

                elapsed_time = time.time() - start_time  # 파일 처리에 걸린 시간 계산
                logging.info(f"File processed and saved: {file_name} (Time taken: {elapsed_time:.2f}s)")

                # 성공적으로 처리된 경우 다음 파일로 이동
                break

            except Exception as error:
                if "429 Too Many Requests" in str(error):
                    logging.warning(f"429 Too Many Requests - Retrying after {delay} seconds.")
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)  # 대기 시간을 점진적으로 증가, 최대 max_delay까지
                else:
                    logging.error(f"Error processing file {file_name}: {error}")
                    break  # 다른 종류의 오류가 발생하면 종료하고 다음 파일로 넘어감

    # 전체 작업 시간 계산 및 출력
    total_elapsed_time = time.time() - total_start_time
    logging.info(f"Total processing time for all files: {total_elapsed_time:.2f} seconds")

