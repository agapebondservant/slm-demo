#!/usr/bin/env python3

from transformers import AutoModelForCausalLM, GPT2Tokenizer

from dotenv import load_dotenv

load_dotenv()

import sys

import os

import torch

repo_name = sys.argv[1]

clone_url = (f"https://{os.getenv('DATA_E2E_HUGGINGFACE_USERNAME')}:"
             f"{os.getenv('DATA_E2E_HUGGINGFACE_TOKEN')}@huggingface.co/"
             f"{os.getenv('DATA_E2E_HUGGINGFACE_USERNAME')}/{repo_name}")

os.system(f"git clone {clone_url}; cd {repo_name}; git lfs install; huggingface-cli lfs-enable-largefiles .")

model_name = f"{os.getenv('DATA_E2E_HUGGINGFACE_USERNAME')}/{repo_name}"

print(f"=====================\nSaving model {model_name}...\n=====================\n")


model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

model.save_pretrained("distilgpt2")

tokenizer.save_pretrained("distilgpt2")

os.system(f"cd {repo_name}; mv ../distilgpt2/* .;"
          "rm -rf ../distilgpt2; git add .;"
          "git commit -m 'Uploaded pretrained model';"
          f"git push; cd -; rm -rf {repo_name}")



