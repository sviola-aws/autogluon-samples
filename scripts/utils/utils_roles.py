import time
import boto3
import json


def create_sns_lambda_role(role_name, sns_topic_arn, sns_policy_name):
    iam = boto3.client("iam")
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "lambda.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
            Description="Role for Lambda to provide S3 read only access",
        )

        role_arn = response["Role"]["Arn"]

        response = iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )

        response = iam.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
            RoleName=role_name,
        )
        
        # Define the policy document
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "VisualEditor0",
                    "Effect": "Allow",
                    "Action": "sns:Publish",
                    "Resource": sns_topic_arn
                }
            ]
        }
        
        response = iam.put_role_policy(
            RoleName=role_name,
            PolicyName=sns_policy_name,
            PolicyDocument=json.dumps(policy_document),
        )

        print("Waiting 30 seconds for the IAM role to propagate")
        time.sleep(30)
        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using ARN from existing role: {role_name}")
        response = iam.get_role(RoleName=role_name)
        return response["Role"]["Arn"]


def create_sagemaker_lambda_role(role_name, sns_topic_arn, sns_policy_name):
    iam = boto3.client("iam")
    try:
        response = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(
                {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "lambda.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
            ),
            Description="Role for Lambda to deploy sagemaker endpoint",
        )

        role_arn = response["Role"]["Arn"]

        response = iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        )

        response = iam.attach_role_policy(
            PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            RoleName=role_name,
        )
        
        # Define the policy document
        policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "VisualEditor0",
                    "Effect": "Allow",
                    "Action": "sns:Publish",
                    "Resource": sns_topic_arn
                }
            ]
        }
        
        response = iam.put_role_policy(
            RoleName=role_name,
            PolicyName=sns_policy_name,
            PolicyDocument=json.dumps(policy_document),
        )

        print("Waiting 30 seconds for the IAM role to propagate")
        time.sleep(30)
        return role_arn

    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Using ARN from existing role: {role_name}")
        response = iam.get_role(RoleName=role_name)
        return response["Role"]["Arn"]
