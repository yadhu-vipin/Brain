import subprocess

def handler(event, context):
    try:
        result = subprocess.run(["python3", "model.py"], capture_output=True, text=True)
        return {
            "statusCode": 200,
            "body": result.stdout,
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": str(e),
        }
