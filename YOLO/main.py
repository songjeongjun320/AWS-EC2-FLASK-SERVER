from datetime import datetime
from dotenv import load_dotenv
import time
import os
import boto3
from yolo import detect  # Assuming detect is a custom module you've implemented
import logging

# Logger
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Setting the Filehandler
log_file = 'log.txt'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Add Handler Logger
LOGGER.addHandler(file_handler)

from typing import List, Tuple

# path for CCTV video files
path: str = ""

# It will return 
def read_cntr_number_region(video_path) -> str:
    # Example weights and configuration
    weight = "./runs/train/TruckNumber_yolov5s_results34/weights/best.pt"
    conf_threshold = 0.5

    # Run detection using your custom detect module
    max_conf_img_path = detect.run(weights=weight, source=video_path, conf_thres=conf_threshold)
    print("Detection Completed at: ", datetime.now())
    print("The most confident img path: ", max_conf_img_path)
    return max_conf_img_path

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

# Extract the text from the cutted image file
def send_to_AWS_Textract(max_conf_img_path) -> List[str]:
    # connect AWS Textract acct
    area, access_id, access_key = get_acct()
    try:
        textract = boto3.client(
            'textract',
            aws_access_key_id=access_id,
            aws_secret_access_key=access_key,
            region_name=area
        )
    except Exception as e:
        print("Can't connect to Textract", e)
    else:
        print("Textract connected!")
    
    # Open the img
    with open(max_conf_img_path, 'rb') as image:
        imageBytes = image.read()

    # Sending img to Textract
    response = textract.detect_document_text(Document={'Bytes': imageBytes})
    print("AWS Textract Response:", response)
    extracted_result = read_result_from_Textract(response)        
    return extracted_result

def read_result_from_Textract(response) -> List[str]:
    print("Image read by Textract: ")
    output: List[str] = []
    # Get the results from here
    if 'Blocks' in response:
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                print(item['Text'])
                output.append(item['Text'])
    else:
        print("No text detected.")
    return output


# Main function to handle the folder or file
def main(video_path: str="C:/Users/frank/Desktop/flask_server/sample.mp4") -> None:
    LOGGER.info(f"Processing single video file: {video_path}")
    read_cntr_number_region(video_path)

if __name__ == "__main__":
    main()