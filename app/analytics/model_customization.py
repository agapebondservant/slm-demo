import datahub.emitter.mce_builder as builder
import datahub.metadata.schema_classes as models
from datahub.emitter.mcp import MetadataChangeProposalWrapper
from datahub.emitter.rest_emitter import DatahubRestEmitter
from huggingface_hub import ModelCard
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
from dotenv import load_dotenv
import os
import mlflow
from mlflow import MlflowClient
from transformers import pipeline
import mlflow.pyfunc
import logging
import traceback
import requests
import json

load_dotenv()


def send_metadata(model_name: str,
                  platform: str,
                  env: str,
                  gms_server: str,
                  model_description: str = None):
    with mlflow.start_run(run_name='send_metadata', nested=True):
        emitter = DatahubRestEmitter(gms_server=gms_server, extra_headers={})
        model_urn = builder.make_ml_model_urn(
            model_name=model_name, platform=platform, env=env
        )
        model_card = ingest_metadata_from_huggingface_model(model_name)

        metadata_change_proposal = MetadataChangeProposalWrapper(
            entityType="mlModel",
            changeType=models.ChangeTypeClass.UPSERT,
            entityUrn=model_urn,
            aspectName="mlModelProperties",
            aspect=models.MLModelPropertiesClass(
                description=model_card.text,
                customProperties={**{k: ','.join(v) for (k, v) in model_card.data.to_dict().items()},
                                  **{'Last Updated': ''}})
    )

    emitter.emit(metadata_change_proposal)


def ingest_metadata_from_huggingface_model(model_name: str):
    card = ModelCard.load(model_name)
    return card or {}


def publish_model(repo_name: str, pretrained_model_name: str):
    with mlflow.start_run(run_name='publish_model', nested=True):
        # TODO: DO NOT HARDCODE!!!
        clone_url = (f"https://tanzuhuggingface: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@huggingface.co/"
                     f"tanzuhuggingface/{repo_name}")

        model_name = f"tanzuhuggingface/{repo_name}"

        print(f"=====================\nSaving model {model_name}...\n=====================\n")

        model = DistilBertTokenizer.from_pretrained(pretrained_model_name)
        tokenizer = TFDistilBertForQuestionAnswering.from_pretrained(pretrained_model_name)
        model.save_pretrained(pretrained_model_name)
        tokenizer.save_pretrained(pretrained_model_name)

        os.system(f"git clone {clone_url}; "
                  f"cd {repo_name};"
                  "git config --global user.email 'tanzuhuggingface@example.com';"
                  "git config --global user.name 'Tanzu Huggingface';"
                  f" git lfs install; "
                  f"huggingface-cli lfs-enable-largefiles .;"
                  f"mv ../{pretrained_model_name}/* .;"
                  f"rm -rf ../{pretrained_model_name}; "
                  "git add .;"
                  "git commit -m 'Uploaded pretrained model';"
                  f"git push; "
                  f"cd -; rm -rf {repo_name}")


def promote_model_to_staging(model_name, pipeline_name):

    # TODO: Determine correct version
    with mlflow.start_run(run_name='promote_model_to_staging', nested=True) as run:
        client = MlflowClient()

        qa_pipe = pipeline(pipeline_name, model_name)
        mlflow.transformers.log_model(
            transformers_model=qa_pipe,
            artifact_path=pipeline_name,
        )

        registered_model_name = model_name.replace('/', '-')
        client.create_registered_model(registered_model_name)
        model_uri = f"runs:/{run.info.run_id}/{pipeline_name}"
        mv = client.create_model_version(registered_model_name, model_uri, run.info.run_id)
        client.transition_model_version_stage(
            name=registered_model_name,
            version=mv.version,
            stage="Staging"
        )


def select_base_llm(prioritized_models: list[str], model_stage: str = 'Production'):
    default_model = _llm_model_name_mappings().get(prioritized_models[-1])

    for registered_model_name in prioritized_models:
        try:
            logging.error("Retrieving production model if exists...")
            model_api_uri = f'{os.getenv("MLFLOW_TRACKING_URI")}/api/2.0/mlflow/registered-models/get?name={registered_model_name}'
            models = requests.get(model_api_uri).json()
            model_key = next((x['name'] for x in models['registered_model']['latest_versions'] if x['current_stage'].lower() == 'production'), None)
            if model_key:
                model_name = _llm_model_name_mappings().get(model_key)
                logging.error(f"Production model found for {registered_model_name}: {model_name}")
                return model_name
        except Exception as e:
            logging.error(f"Model name={registered_model_name}, stage={model_stage} not found.")
            logging.info(str(e))
            logging.info(''.join(traceback.TracebackException.from_exception(e).format()))
            
    return default_model


# TODO: Do not hardcode mappings!!!
def _llm_model_name_mappings():
    return {
        'tanzuhuggingface-open-llama-7b-open-instruct-GGML': 'TheBloke/open-llama-7b-open-instruct-GGML',
        'tanzuhuggingface-testrepo': 'tanzuhuggingface/dev'
    }

