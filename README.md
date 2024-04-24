# Small Language Models with Tanzu

## Contents
1. [Set up HuggingFace model repo](#huggingfacerepo)

### Set up HuggingFace model repo<a name="huggingfacerepo"/>
#### Prequisites:
- [ ] git-lfs
- [ ] Set up an SSH Key <DATA_E2E_HUGGINGFACE_TOKEN> for account <DATA_E2E_HUGGINGFACE_USERNAME>: <a href="https://huggingface.co/settings/keys" target="_blank">link</a>

1. Deploy required python dependencies:
```
pip install -r requirements.txt
```

2. Set up SLM repo:
```
source .env
export REPO_NAME=tanzu #your repo name
export SLM_NAME=microsoft/Phi-3-mini-4k-instruct-onnx
resources/scripts/create-huggingface-model-repo.sh $REPO_NAME
```

2. Publish a model to the SLM repo:
```
$(which python3)  -c "from app.analytics import slm; slm.publish_slm(repo_name='${REPO_NAME}', pretrained_model_name='${SLM_NAME}')"
```

3. Publish VMware Llama-7B model to the repo:
```
export VMWARE_LLAMA_REPO_NAME=<your repo name for vmware llama model>
resources/scripts/create-huggingface-model-repo.sh $VMWARE_LLAMA_REPO_NAME
resources/scripts/save-dummy-huggingface-model-text-generation.sh $VMWARE_LLAMA_REPO_NAME
$(which python3)  -c "from app.analytics import model_customization; import os; \
                        os.environ['MLFLOW_EXPERIMENT_NAME']='question_answering_model_team_main'; \
                        model_customization.promote_model_to_staging('tanzuhuggingface/open-llama-7b-open-instruct-GGML', 'text-generation')"
```