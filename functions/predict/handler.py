import os
import subprocess
import sys

def handler(event, context):
    python_executable = os.getenv('PYTHON_PATH', 'python3')
    try:
        # Log the current working directory and file paths
        print(f"Using Python executable: {python_executable}", file=sys.stderr)
        print(f"Current working directory: {os.getcwd()}", file=sys.stderr)
        
        model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
        print(f"Model file path: {model_path}", file=sys.stderr)
        
        # Run your model.py with the correct paths
        result = subprocess.run([python_executable, 'model.py', model_path], capture_output=True, text=True)
        print(f"Python script output: {result.stdout}", file=sys.stderr)
        return {
            "statusCode": 200,
            "body": result.stdout,
        }
    except FileNotFoundError as fnf_error:
        return {
            "statusCode": 500,
            "body": f"File not found error: {fnf_error}",
        }
    except subprocess.CalledProcessError as cpe_error:
        return {
            "statusCode": 500,
            "body": f"Called process error: {cpe_error}",
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Exception: {e}",
        }
