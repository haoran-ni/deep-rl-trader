import boto3
from botocore.config import Config
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, DATA_ROOT_DIR, TRAIN_START_DATE, TRADE_END_DATE
import os
from botocore.exceptions import ClientError
import pandas_market_calendars as mcal

# Find all trading days

# Date range (adjust start_date as needed)
start_date = "2015-12-01"
end_date = "2025-11-26"

# Get NASDAQ trading schedule and extract trading days
nasdaq = mcal.get_calendar("NASDAQ")
schedule = nasdaq.schedule(start_date=start_date, end_date=end_date)
trading_days = [d.strftime("%Y-%m-%d") for d in schedule.index.date]

# Optional: build S3 object keys matching your path pattern
object_keys = [
    f"flatfiles/us_stocks_sip/minute_aggs_v1/{d[:4]}/{d[5:7]}/{d}.csv.gz"
    for d in trading_days
]

# Initialize a session using your credentials
session = boto3.Session(
  aws_access_key_id=AWS_ACCESS_KEY,
  aws_secret_access_key=AWS_SECRET_KEY,
)

# Create a client with your session and specify the endpoint
s3 = session.client(
  's3',
  endpoint_url='https://files.massive.com',
  config=Config(signature_version='s3v4'),
)

# Specify the bucket name
bucket_name = 'flatfiles'

# Specify the S3 object key name
try:
    for object_key in object_keys:
        # Remove the bucket name (e.g. 'flatfiles/') prefix if present in object_key
        if object_key.startswith(bucket_name + '/'):
            object_key = object_key[len(bucket_name + '/'):]

        # Specify the local file name and path to save the downloaded file
        local_file_name = object_key.split('/')[-1]  # e.g., '2025-06-12.csv.gz'
        local_file_path = DATA_ROOT_DIR + '/massive_dataset/' + local_file_name

        dirpath = os.path.dirname(local_file_path) or '.'
        os.makedirs(dirpath, exist_ok=True)

        try:
            if os.path.exists(local_file_path):
                print(f"File {local_file_path} already exists. Skipping download.")
                continue
            s3.download_file(bucket_name, object_key, local_file_path)
            print(f"Downloaded s3://{bucket_name}/{object_key} -> {local_file_path}")
        except ClientError as e:
            err_code = e.response.get("Error", {}).get("Code", "")
            if err_code == "404":
                print("The requested object does not exist.")
            else:
                raise
except Exception as e:
    print(f"An error occurred: {e}")
    print("Download interrupted. Saving files downloaded so far.")