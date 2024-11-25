from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
import yolo.detect
import logging
import random
import shutil
import boto3
import sys
import re
import os

# Logger
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


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

def get_acct():
    area = os.getenv('TEXTRACT_AREA')
    access_id = os.getenv('ACCESS_ID')
    access_key = os.getenv('ACCESS_KEY')
    print(area, access_id, access_key)
    return area, access_id, access_key


# It will return extracted result
def read_cntr_number_region(video_path, driver_name) -> str:
    # Example weights and configuration
    weight = "./runs/train/TruckNumber_yolov5s_results34/weights/best.pt"
    conf_threshold = 0.5

    # Run detection using your custom detect module
    max_conf_img_path = yolo.detect.run(weights=weight, source=video_path, conf_thres=conf_threshold)
    print("Detection Completed at: ", datetime.now())
    print("The most confident img path: ", max_conf_img_path)

    logging.info(f"LOG-- Most confident image path: {max_conf_img_path}")
    extracted_result = send_to_AWS_Textract(max_conf_img_path, driver_name)
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
        print("read_result_from_Textract(response): ", extracted_result)

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

        upload_and_cleanup_videos(new_id)
        # Return the extracted result (processed text, not a file path)
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
    logging.info("Starting image upload to Supabase storage...")
    client = createSupabaseClient_YMS()
    storage_bucket = os.getenv("STORAGE_IMG_BUCKET")  # Supabase Storage 버킷 이름
    file_name = f"{new_id:03}.jpg"  # 파일 이름은 row의 id 기반으로 설정

    logging.info(f"Target bucket: {storage_bucket}, file name: {file_name}")

    try:
        # 이미지 파일 읽기
        logging.info(f"Reading image file: {max_conf_img_path}")
        with open(max_conf_img_path, "rb") as file:
            file_content = file.read()  # 파일 내용을 읽기
            response = client.storage.from_(storage_bucket).upload(file_name, file_content)

        # 응답 확인: response가 성공적으로 반환되었는지 확인
        if not response.path:
            raise Exception(f"Upload failed, no path returned. Response: {response}")

        # 업로드 성공 로그
        logging.info(f"Image uploaded successfully: {file_name} to bucket {storage_bucket}")
        logging.info(f"Uploaded file path: {response.full_path}")

    except FileNotFoundError:
        logging.error(f"Image file not found: {max_conf_img_path}")
        raise Exception(f"Image file not found: {max_conf_img_path}")
    except Exception as e:
        logging.error(f"Error uploading image to Supabase storage: {e}")
        raise Exception(f"Failed to upload image to Supabase storage: {e}")


def upload_and_cleanup_videos(new_id):
    # Supabase 클라이언트 생성
    supabase_yms = createSupabaseClient_YMS()

    # runs/detect/exp 디렉토리 설정
    detect_dir = os.path.join(os.getcwd(), "yolo", "runs", "detect")
    exp_dir = os.path.join(detect_dir, "exp")
    
    # .mp4 파일 업로드
    if os.path.exists(exp_dir):
        mp4_files = [f for f in os.listdir(exp_dir) if f.endswith('.mp4')]
        logging.info(f"LOG-- DETECTED MP4 FILE: {mp4_files}")
        
        if mp4_files:
            for mp4_file in mp4_files:
                file_path = os.path.join(exp_dir, mp4_file)
                try:
                    with open(file_path, "rb") as video_file:
                        storage_bucket = os.getenv("STORAGE_RESULT_VIDEO_BUCKET")
                        # Supabase에 파일 업로드
                        logging.info(f"LOG-- Attempting to upload {mp4_file} to bucket {storage_bucket}")

                        response = supabase_yms.storage.from_(storage_bucket).upload(
                            f"{new_id}.mp4", video_file
                        )

                        if not response.path:
                            raise Exception(f"Upload failed, no path returned. Response: {response}")

                        # 업로드 성공 로그
                        logging.info(f"Image uploaded successfully: {new_id}.mp4 to bucket {storage_bucket}")
                        logging.info(f"Uploaded file path: {response.full_path}")

                except Exception as e:
                    logging.error(f"LOG-- Error uploading {mp4_file}: {e}")
        else:
            logging.warning(f"LOG-- No .mp4 files found in {exp_dir}.")
    else:
        logging.warning(f"LOG-- Directory {exp_dir} does not exist, skipping upload.")

    # runs/detect 디렉토리 내 모든 하위 폴더 삭제
    if os.path.exists(detect_dir):
        for folder in os.listdir(detect_dir):
            folder_path = os.path.join(detect_dir, folder)
            if os.path.isdir(folder_path):
                shutil.rmtree(folder_path)  # 폴더 삭제
                logging.info(f"LOG-- Deleted folder: {folder_path}")
    else:
        logging.warning(f"LOG-- Directory {detect_dir} does not exist, skipping cleanup.")


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