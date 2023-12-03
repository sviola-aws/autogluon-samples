from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)

import sagemaker
import sys

from typing import Dict

from sagemaker import image_uris, model_uris, script_uris, hyperparameters

from sagemaker.tensorflow import TensorFlow
from sagemaker.sklearn.processing import ScriptProcessor

from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.lambda_helper import Lambda

from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.functions import JsonGet

from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import Join

from sagemaker.inputs import TrainingInput

from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model import Model
from sagemaker.estimator import Estimator

from sagemaker.workflow.pipeline import Pipeline

sys.path.append("scripts/utils")
from utils_roles import create_sns_lambda_role, create_sagemaker_lambda_role
from sagemaker.workflow.pipeline_context import PipelineSession


def get_pipeline(config_dict: Dict):
    model_package_name = config_dict["MODEL_PACKAGE_NAME"]

    # input data parameters
    input_data = ParameterString(name="InputDataPath", default_value=config_dict["INPUT_PATH"])

    # preprocessing parameters
    preprocessing_instance_type = ParameterString(
        name="ProcessingInstanceType",
        default_value=config_dict["PREPROCESSING_STEP_CONFIG"]["instance_type"],
    )

    preprocessing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount",
        default_value=config_dict["PREPROCESSING_STEP_CONFIG"]["instance_count"],
    )

    preprocessing_framework_version = ParameterString(
        name="ProcessingFrameworkVersion",
        default_value=config_dict["PREPROCESSING_STEP_CONFIG"]["framework_version"],
    )

    # training parameters
    training_instance_type = ParameterString(
        name="TrainingInstanceType",
        default_value=config_dict["MODEL_TRAINING_STEP_CONFIG"]["instance_type"],
    )

    training_instance_count = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=config_dict["MODEL_TRAINING_STEP_CONFIG"]["instance_count"],
    )

    training_python_version = ParameterString(
        name="TrainingPythonVersion",
        default_value=config_dict["MODEL_TRAINING_STEP_CONFIG"]["python_version"],
    )

    # evaluation parameters
    evaluation_instance_type = ParameterString(
        name="EvaluationInstanceType",
        default_value=config_dict["MODEL_EVALUATION_STEP_CONFIG"]["instance_type"],
    )

    evaluation_instance_count = ParameterInteger(
        name="EvaluationInstanceCount",
        default_value=config_dict["MODEL_EVALUATION_STEP_CONFIG"]["instance_count"],
    )

    # model creation parameters
    model_creation_instance_type = ParameterString(
        name="ModelCreationInstanceType",
        default_value=config_dict["CREATE_MODEL_CONFIG"]["instance_type"],
    )
    
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
    )
    
    # ----------------------------------------------------------------
    # pre-processing step
    # ----------------------------------------------------------------    
    sklearn_processor = SKLearnProcessor(
        framework_version=preprocessing_framework_version,
        role=config_dict["ROLE"],
        instance_type=preprocessing_instance_type,
        instance_count=preprocessing_instance_count,
        base_job_name=f"{model_package_name}-processing",
    )

    step_processing = ProcessingStep(
        name="PreprocessStep",
        processor=sklearn_processor,
        outputs=[
            ProcessingOutput(
                output_name="train_data", 
                source="/opt/ml/processing/train", 
                s3_upload_mode="EndOfJob", 
                destination=Join(on="/",
                                 values=[config_dict["MODEL_SAVE_PATH"],
                                         "datasets",
                                         "train"])
            ),
            ProcessingOutput(
                output_name="test_data", 
                source="/opt/ml/processing/test",
                s3_upload_mode="EndOfJob", 
                destination=Join(on="/",
                                 values=[config_dict["MODEL_SAVE_PATH"],
                                         "datasets",
                                         "test"])
            ),
        ],
        code="scripts/processing/processing.py",
        job_arguments=["--input-data", input_data, 
                       "--transformation-path", config_dict["MODEL_SAVE_PATH"]],
    )
   
    # ----------------------------------------------------------------
    # model training step
    # ----------------------------------------------------------------    
    
    train_model_id, train_model_version, train_scope = "autogluon-classification-ensemble", "*", "training"
    
    # Retrieve the docker image
    training_instance_type = "ml.p3.2xlarge"

    # Retrieve the docker image
    image_uri = image_uris.retrieve(
        region=None,
        framework=None,
        model_id=train_model_id,
        model_version=train_model_version,
        image_scope=train_scope,
        instance_type=training_instance_type
    )
    
    # Retrieve the training script
    train_source_uri = script_uris.retrieve(
        model_id=train_model_id, model_version=train_model_version, script_scope=train_scope
    )
    # Retrieve the pre-trained model tarball to further fine-tune. In tabular case, however, the pre-trained model tarball is dummy and fine-tune means training from scratch.
    train_model_uri = model_uris.retrieve(
        model_id=train_model_id, model_version=train_model_version, model_scope=train_scope
    )
    
    # Retrieve the default hyper-parameters for fine-tuning the model
    hyperparams = hyperparameters.retrieve_default(
        model_id=train_model_id, model_version=train_model_version
    )

    # [Optional] Override default hyperparameters with custom values
    hyperparams["auto_stack"] = "True"
    print(hyperparams)
    
    training_estimator = Estimator(
        image_uri=image_uri,
        source_dir=train_source_uri,
        model_uri=train_model_uri,
        entry_point="transfer_learning.py",
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        output_path=Join(on="/", 
                         values=[config_dict["MODEL_SAVE_PATH"], 
                                 "model_artifacts"]),
        role=config_dict["ROLE"],
        sagemaker_session=PipelineSession(),
        max_run=360000,
        hyperparameters=hyperparams,
    )

    step_model_training = TrainingStep(
        name="TrainingStep",
        estimator=training_estimator,
        inputs={
            "training": TrainingInput(
                s3_data=step_processing.properties.ProcessingOutputConfig.Outputs[
                    "train_data"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_processing.properties.ProcessingOutputConfig.Outputs[
                    "test_data"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # ----------------------------------------------------------------
    # model registration step
    # ----------------------------------------------------------------

    step_register_model = RegisterModel(
        name="RegisterStep",
        estimator=training_estimator,
        model_data=step_model_training.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_name,
        approval_status=model_approval_status,
    )

    # ----------------------------------------------------------------
    # model creation step
    # ----------------------------------------------------------------
    model = Model(
        image_uri=image_uri,
        model_data=step_model_training.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=PipelineSession(),
        role=config_dict["ROLE"],
    )
    
    step_create_model = ModelStep(
        name="CreateModelStep",
        step_args=model.create(instance_type="ml.m5.xlarge"),
        depends_on=["RegisterStep"],
    )


    # creating whole pipeline

    pipeline = Pipeline(
        name=config_dict["PIPELINE_NAME"],
        parameters=[
            input_data,
            preprocessing_instance_type,
            preprocessing_instance_count,
            preprocessing_framework_version,
            training_instance_type,
            training_instance_count,
            training_python_version,
            model_creation_instance_type,
            model_approval_status,
        ],
        steps=[step_processing, step_model_training, step_register_model, step_create_model],
    )

    return pipeline
