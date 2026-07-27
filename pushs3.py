
import os
import shutil
import subprocess
import pandas as pd
import requests
from huggingface_hub import HfApi, hf_hub_download
from google.colab import userdata
import boto3

os.environ['AWS_ACCESS_KEY_ID'] = userdata.get('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = userdata.get('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

BUCKET = "nickb-aarj" 
s3 = boto3.client('s3')
api = HfApi()

TMP_DIR = "./tmp_download"
FAILED_LOG = "./failed_downloads.csv"


def file_exists_in_s3(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def stream_repo_to_s3(repo_id, s3_prefix, repo_type="dataset"):
    print(f"\n=== {repo_id} ===")
    try:
        files = api.list_repo_files(repo_id, repo_type=repo_type)
    except Exception as e:
        print(f"ERROR listing files: {e}")
        return

    for filename in files:
        s3_key = f"{s3_prefix}/{filename}"
        if file_exists_in_s3(BUCKET, s3_key):
            print(f"Skipping (already uploaded): {filename}")
            continue
        try:
            print(f"Downloading: {filename}")
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=TMP_DIR)
            s3.upload_file(local_path, BUCKET, s3_key)
            os.remove(local_path)
        except Exception as e:
            print(f"ERROR on {filename}: {e}")
            continue

    print(f"Done: {repo_id}")


def stream_url_dataset_to_s3(repo_id, s3_prefix, url_column="video_url"):
    print(f"\n=== {repo_id} ===")
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        print(f"ERROR listing files: {e}")
        return

    data_files = [f for f in files if f.endswith((".csv", ".parquet", ".json", ".jsonl"))]
    if not data_files:
        print("No metadata file found.")
        return

    metadata_records = []
    for data_file in data_files:
        local_path = hf_hub_download(repo_id=repo_id, filename=data_file, repo_type="dataset", local_dir=TMP_DIR)

        meta_s3_key = f"{s3_prefix}/_metadata/{data_file}"
        if not file_exists_in_s3(BUCKET, meta_s3_key):
            s3.upload_file(local_path, BUCKET, meta_s3_key)

        try:
            if data_file.endswith(".csv"):
                df = pd.read_csv(local_path, on_bad_lines="skip", encoding="utf-8", encoding_errors="replace")
            elif data_file.endswith(".parquet"):
                df = pd.read_parquet(local_path)
            else:
                df = pd.read_json(local_path, lines=data_file.endswith(".jsonl"))
        except Exception as e:
            print(f"ERROR reading {data_file}: {e}")
            continue

        if url_column not in df.columns:
            print(f"'{url_column}' not found. Columns: {list(df.columns)}")
            continue

        metadata_records.append(df)
        os.remove(local_path)

    if not metadata_records:
        print("No usable metadata found.")
        return

    full_df = pd.concat(metadata_records, ignore_index=True)
    print(f"Found {len(full_df)} video URLs.")

    failed_rows = []
    for idx, row in full_df.iterrows():
        video_url = row[url_column]
        if pd.isna(video_url):
            continue

        filename = video_url.split("/")[-1].split("?")[0]
        s3_key = f"{s3_prefix}/videos/{filename}"
        if file_exists_in_s3(BUCKET, s3_key):
            continue

        local_path = os.path.join(TMP_DIR, filename)
        try:
            resp = requests.get(video_url, timeout=30, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            s3.upload_file(local_path, BUCKET, s3_key)
            os.remove(local_path)
            if idx % 50 == 0:
                print(f"[{idx}/{len(full_df)}] {filename}")
        except Exception as e:
            print(f"FAILED ({idx}): {video_url} -> {e}")
            failed_rows.append({"index": idx, "video_url": video_url, "error": str(e)})
            if os.path.exists(local_path):
                os.remove(local_path)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(FAILED_LOG, mode="a", index=False, header=not os.path.exists(FAILED_LOG))
        print(f"{len(failed_rows)} failed. Logged to {FAILED_LOG}")

    print(f"Done: {repo_id} ({len(full_df) - len(failed_rows)}/{len(full_df)} uploaded)")


def push_github_repo_to_s3(github_url, s3_prefix):
    print(f"\n=== {github_url} ===")
    local_dir = "./tmp_github_repo"
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)

    subprocess.run(["git", "clone", github_url, local_dir], check=True)

    for root, dirs, files in os.walk(local_dir):
        if ".git" in root:
            continue
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_dir)
            s3_key = f"{s3_prefix}/{relative_path}"
            if file_exists_in_s3(BUCKET, s3_key):
                continue
            s3.upload_file(local_file, BUCKET, s3_key)

    shutil.rmtree(local_dir)
    print(f"Done: {github_url}")


stream_repo_to_s3("INSAIT-Institute/ImplausiBench", "datasets/implausibench", repo_type="dataset")

stream_url_dataset_to_s3("videophysics/videophy2_test", "datasets/videophy2_test", url_column="video_url")
stream_url_dataset_to_s3("videophysics/videophy2_train", "datasets/videophy2_train", url_column="video_url")

stream_repo_to_s3("Efficient-Large-Model/vila-ewm-qwen2-1.5b", "models/vila-ewm-qwen2-1.5b", repo_type="model")
stream_repo_to_s3("videophysics/videophy_2_auto", "models/videophy_2_auto", repo_type="model")
stream_repo_to_s3("NU-World-Model-Embodied-AI/phyjudge-9B", "models/phyjudge-9B", repo_type="model")
stream_repo_to_s3("Qwen/Qwen2.5-VL-7B-Instruct", "models/qwen2.5-vl-7b-instruct", repo_type="model")
stream_repo_to_s3("OpenGVLab/InternVL2_5-8B", "models/internvl2.5-8b", repo_type="model")
stream_repo_to_s3("llava-hf/llava-onevision-qwen2-7b-ov-hf", "models/llava-onevision-7b", repo_type="model")

push_github_repo_to_s3("https://github.com/Hritikbansal/videophy.git", "code/videophy")

if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)

print("\nDone. Everything is in S3.")
if os.path.exists(FAILED_LOG):
    print(f"Some downloads failed — check {FAILED_LOG}")