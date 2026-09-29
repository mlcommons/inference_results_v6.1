## Scripts to reach files in S3 bucket

### Setup

Install the required packages (recommended to use a separate Python environment, e.g. virtualenv):

`pip install -r requirements.txt`

Also need to request for the proper credentials, name of the bucket, etc.

### Upload / delete
```bash
python3 upload_or_delete_dir.py
    -b <bucket-name>
    -i <local_dir_to_upload>
    -a <aws_access_key_id>
    -s <aws_secret_access_key>
    -r <region_name>
    -e <endpoint_url>
    -p <remote_path>
    [--delete] # just delete files in the bucket
    [--overwrite] # delete files in the bucket, then upload the files from local directory
```

The content of a local directory (`-i` / `--input_dir`) will be uploaded into the S3 bucket specified by `--bucket` + `--remote_path`.

### Download / list
```bash
python3 download_or_list_dir.py
    -b <bucket-name>
    -o <local_dir_to_save_files>
    -a <aws_access_key_id>
    -s <aws_secret_access_key>
    -r <region_name>
    -e <endpoint_url>
    -p <remote_path>
    [-l] # just list files in the bucket
    [--overwrite] # delete the local directory, then download the files from the bucket
```

The files in the specified S3 bucket (`--bucket` + `--remote_path`) will be downloaded into the local directory specified by `-o` / `--output_dir`.

### Note

If the bucket name is not terminated by `/` then all of the files in the bucket with the same prefix will be selected.

Example:
```bash
# Bucket name is not terminated by '/'
$ python3 download_or_list_dir.py -b s3://bucket/inference -l -a ...
inference/accuracy_logs/1/log_accuracy.json
inference/accuracy_logs/2/log_accuracy.json
inference2/accuracy_logs/1/log_accuracy.json
inference2/accuracy_logs/2/log_accuracy.json
inference3/accuracy_logs/1/log_accuracy.json
inference3/accuracy_logs/2/log_accuracy.json

# Bucket name is not terminated by '/'
$ python3 download_or_list_dir.py -b s3://bucket/inference/ -l -a ...
inference/accuracy_logs/1/log_accuracy.json
inference/accuracy_logs/2/log_accuracy.json
```

### File-level download, upload and delete

```bash
python3 single_file_operations.py
    -b <bucket-name>
    -a <aws_access_key_id>
    -s <aws_secret_access_key>
    -r <region_name>
    -e <endpoint_url>
    {download,upload,delete}
    --remote-file <file_path_in_bucket>
    [--local-file <file_path_locally]
```

Example:
```bash
# $ python3 download_or_list_dir.py -b s3://bucket/inference/ -l -a ...
# inference/accuracy_logs/2/log_accuracy.json

# - Download
$ python3 single_file_operations.py -b s3://bucket/inference download --remote-file inference/accuracy_logs/2/log_accuracy.json --local-file ./best_accuracy.json

# $ ls
# best_accuracy.json

# - Upload
$ python3 single_file_operations.py -b s3://bucket/inference upload --local-file ./best_accuracy.json --remote-file inference/accuracy_logs/submission/final/log_accuracy.json

# $ python3 download_or_list_dir.py -b s3://bucket/inference/ -l -a ...
# inference/accuracy_logs/2/log_accuracy.json
# inference/accuracy_logs/submission/final/log_accuracy.json

# - Delete
$ python3 single_file_operations.py -b s3://bucket/inference delete --remote-file inference/accuracy_logs/2/log_accuracy.json

# $ python3 download_or_list_dir.py -b s3://bucket/inference/ -l -a ...
# inference/accuracy_logs/submission/final/log_accuracy.json
```
