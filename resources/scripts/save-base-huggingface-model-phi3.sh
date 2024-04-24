#!/usr/bin/env python3

from transformers import BartForConditionalGeneration, BartTokenizer

from dotenv import load_dotenv

load_dotenv()

import sys

import os

repo_name = sys.argv[1]

clone_url = (f"https://{os.getenv('DATA_E2E_HUGGINGFACE_USERNAME')}:"
             f"{os.getenv('DATA_E2E_HUGGINGFACE_TOKEN')}@huggingface.co/"
             f"{os.getenv('DATA_E2E_HUGGINGFACE_USERNAME')}/{repo_name}")

os.system(f"git clone {clone_url}; cd {repo_name}; git lfs install; huggingface-cli lfs-enable-largefiles .")

model_name = f"{os.getenv('DATA_E2E_HUGGINGFACE_USERNAME')}/{repo_name}"

print(f"=====================\nSaving model {model_name}...\n=====================\n")

model = BartForConditionalGeneration.from_pretrained('microsoft/Phi-3-mini-4k-instruct-onnx')

tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

model.save_pretrained('sshleifer/distilbart-cnn-12-6')

tokenizer.save_pretrained('sshleifer/distilbart-cnn-12-6')

os.system(f"cd {repo_name}; mv ../sshleifer/distilbart-cnn-12-6/* .;"
          "rm -rf ../sshleifer; git add .;"
          "git commit -m 'Uploaded pretrained model';"
          f"git push; cd -; rm -rf {repo_name}")



