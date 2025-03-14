import os
import subprocess

def handler(event, context):
    python_executable = os.getenv('PYTHON_PATH', 'python3')
    try:
        # Get the current directory where the handler.py is running
        current_directory = os.path.dirname(__file__)
        # Construct the path to model.pth
        model_path = os.path.join(current_directory, 'model.pth')

        # Log the paths for debugging
        print(f"Using Python executable: {python_executable}")
        print(f"Model file path: {model_path}")
        
        # Run your model.py with the correct paths
        result = subprocess.run([python_executable, 'model.py', model_path], capture_output=True, text=True)
        return {
            "statusCode": 200,
            "body": result.stdout,
        }
    except FileNotFoundError as fnf_error:
        return {
            "statusCode": 500,
            "body": f"File not found error: {fnf_error}",
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Exception: {e}",
        }
