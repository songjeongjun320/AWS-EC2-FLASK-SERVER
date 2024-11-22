from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Tuple
import detect # Assuming detect is a custom module you've implemented
import logging
import random
import boto3
import sys
import re
import os

# Logger
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolo')))

# Setting the Filehandler
log_file = 'log.txt'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Add Handler Logger
LOGGER.addHandler(file_handler)

# path for CCTV video files
path: str = ""

# AWS Textrac #
def configure() -> None:
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    load_dotenv(dotenv_path)

def get_acct() -> Tuple[str, str, str]:
    area = os.getenv('TEXTRACT_AREA')
    access_id = os.getenv('ACCESS_ID')
    access_key = os.getenv('ACCESS_KEY')
    print(area, access_id, access_key)
    return area, access_id, access_key


# It will return extracted result
def read_cntr_number_region(video_path) -> str:
    # Example weights and configuration
    weight = "./runs/train/TruckNumber_yolov5s_results34/weights/best.pt"
    conf_threshold = 0.5

    # Run detection using your custom detect module
    max_conf_img_path = detect.run(weights=weight, source=video_path, conf_thres=conf_threshold)
    print("Detection Completed at: ", datetime.now())
    print("The most confident img path: ", max_conf_img_path)

    logging.info(f"LOG-- Most confident image path: {max_conf_img_path}")
    extracted_result = send_to_AWS_Textract(max_conf_img_path, driver_name="")
    return extracted_result


# Extract the text from the cutted image file
def send_to_AWS_Textract(max_conf_img_path, driver_name) -> str:
    try:
        # Connect to AWS Textract
        area, access_id, access_key = get_acct()
        try:
            textract = boto3.client(
                'textract',
                aws_access_key_id=access_id,
                aws_secret_access_key=access_key,
                region_name=area
            )
        except Exception as e:
            raise Exception(f"Failed to connect to AWS Textract: {e}")
        else:
            print("Textract connected!")
        
        # Open the image
        with open(max_conf_img_path, 'rb') as image:
            imageBytes = image.read()

        # Send image to Textract
        response = textract.detect_document_text(Document={'Bytes': imageBytes})
        print("AWS Textract Response:", response)
        extracted_result = read_result_from_Textract(response)

        # Add new row to Supabase table
        try:
            new_id = postNewRowToSupabase(extracted_result, driver_name)
        except Exception as e:
            raise Exception(f"Failed to add new row to Supabase: {e}")

        # Add new image with row's ID to Supabase storage
        try:
            postNewImgToSupabase(max_conf_img_path, new_id)
        except Exception as e:
            raise Exception(f"Failed to upload image to Supabase storage: {e}")

        # If all processes succeed, return the extracted result
        return extracted_result

    except Exception as e:
        # Log the error and re-raise it for higher-level handling
        print(f"Error in send_to_AWS_Textract: {e}")
        raise

# Send the img to AWS Textract and get the result
def read_result_from_Textract(response)-> str:
    print("Image read by Textract: ")
    result: str = ""
    # Get the results from here
    if 'Blocks' in response:
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                print(item['Text'])
                result += item['Text']
    else:
        print("No text detected.")

    result = re.sub(r'[^A-Za-z0-9]', '', result)
    
    return result

# The new row will be added to Supbase
def postNewRowToSupabase(extracted_result, driver_name):
    client = createSupabaseClient_YMS()

    # 추출 결과에서 컨테이너 번호와 크기 분리
    cntr_num = extracted_result[:11]  # 처음 11글자: 컨테이너 번호
    cntr_size = extracted_result[11:]  # 나머지: 컨테이너 크기

    # 현재 날짜와 시간
    current_time = datetime.now()
    date = current_time.strftime("%Y-%m-%d")  # YYYY-MM-DD 포맷
    time = current_time.strftime("%H:%M:%S")  # HH:MM:SS 포맷

    # in_out은 랜덤으로 설정
    in_out = random.choice([True, False])

    # Supabase에 데이터 삽입
    response = client.table("main").insert({
        "cntr_number": cntr_num,
        "date": date,
        "time": time,
        "in_out": in_out,
        "cntr_size": cntr_size,
        "driver_name": driver_name
    }).execute()

    # 응답 확인
    if response.data:
        logging.info(f"New row added successfully: {response.data}")
        # 새로 추가된 행의 ID 반환
        return response.data[0]['id']  # Supabase 응답에서 첫 번째 행의 'id'를 반환
    elif response.error:
        logging.error(f"Error adding new row: {response.error}")
        raise Exception(f"Supabase Insert Error: {response.error.message}")
    else:
        logging.error("Unknown error occurred while adding new row.")
        raise Exception("Unknown error occurred in Supabase Insert.")

    
def postNewImgToSupabase(max_conf_img_path, new_id):
    client = createSupabaseClient_YMS()
    storage_bucket = "cntr_imgs"  # Supabase Storage 버킷 이름
    file_name = f"{new_id}.jpg"  # 파일 이름은 row의 id 기반으로 설정

    try:
        # 이미지 파일 읽기
        with open(max_conf_img_path, "rb") as file:
            response = client.storage().from_(storage_bucket).upload(file_name, file)

        # 응답 확인: Supabase 클라이언트에서 제공하는 response.error를 사용
        if response.error:
            raise Exception(f"Upload error: {response.error.message}")

        # 업로드 성공 로그
        logging.info(f"LOG-- Image uploaded successfully: {file_name}")

    except Exception as e:
        logging.error(f"Error uploading image to Supabase storage: {e}")
        raise Exception(f"Failed to upload image to Supabase storage: {e}")



# Main function to handle the folder or file
def main(video_path: str="C:/Users/frank/Desktop/flask_server/sample.mp4") -> None:
    LOGGER.info(f"Processing single video file: {video_path}")
    read_cntr_number_region(video_path)


def createSupabaseClient_YMS() -> Client:
    supabase_url = os.getenv("SUPABASE_URL_YMS")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_YMS")
    
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase URL or Anon Key is missing. Check your environment variables.")
    
    supabase: Client = create_client(supabase_url, supabase_key)
    return supabase


if __name__ == "__main__":
    main()