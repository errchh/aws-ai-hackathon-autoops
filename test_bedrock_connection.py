#!/usr/bin/env python3
"""Test script to verify Amazon Bedrock integration."""

import json
import boto3
from config.settings import get_settings

def test_bedrock_connection():
    """Test Bedrock connection and model access."""
    settings = get_settings()
    
    # Test Bedrock Runtime client
    try:
        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=settings.aws.region,
            aws_access_key_id=settings.aws.access_key_id,
            aws_secret_access_key=settings.aws.secret_access_key,
            aws_session_token=settings.aws.session_token
        )
        
        # Test embedding model
        print("Testing Amazon Titan Embed Text v1...")
        body = json.dumps({"inputText": "Hello, this is a test."})
        
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        embedding = response_body['embedding']
        print(f"✅ Embedding generated successfully! Vector length: {len(embedding)}")
        
        # Test Claude model
        print(f"\nTesting {settings.bedrock.model_id}...")
        claude_body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Please respond with 'Bedrock connection successful!'"
                }
            ]
        })
        
        claude_response = bedrock_client.invoke_model(
            modelId=settings.bedrock.model_id,
            body=claude_body,
            contentType="application/json",
            accept="application/json"
        )
        
        claude_response_body = json.loads(claude_response['body'].read())
        message = claude_response_body['content'][0]['text']
        print(f"✅ Claude response: {message}")
        
        print("\n🎉 All Bedrock models are working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing Bedrock connection: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your AWS credentials in .env file")
        print("2. Verify your AWS region has Bedrock access")
        print("3. Ensure model access is enabled in Bedrock console")
        print("4. Check IAM permissions for Bedrock")

if __name__ == "__main__":
    test_bedrock_connection()