const { spawn } = require("child_process");
const path = require("path");

exports.handler = async function (event, context) {
  if (event.httpMethod !== "POST") {
    return {
      statusCode: 405,
      body: JSON.stringify({ error: "Method Not Allowed" }),
    };
  }

  try {
    const body = JSON.parse(event.body);
    const imageBase64 = body.image;

    if (!imageBase64) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: "No image provided" }),
      };
    }

    // Ensure the path works correctly on Netlify
    const pythonScriptPath = path.resolve(__dirname, "predict.py");
    console.log("Using Python script at:", pythonScriptPath);

    const result = await runPythonScript(pythonScriptPath, imageBase64);

    console.log("Python script output:", result);

    const parsedResult = JSON.parse(result);

    return {
      statusCode: 200,
      body: JSON.stringify(parsedResult),
    };
  } catch (error) {
    console.error("Error:", error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message }),
    };
  }
};

function runPythonScript(scriptPath, imageBase64) {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn("python", [scriptPath]);

    let result = "";
    let error = "";

    pythonProcess.stdin.write(imageBase64);
    pythonProcess.stdin.end();

    pythonProcess.stdout.on("data", (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      error += data.toString();
    });

    pythonProcess.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Python script exited with code ${code}: ${error}`));
      } else {
        try {
          resolve(result.trim());
        } catch (e) {
          reject(new Error("Failed to parse output: " + e.message));
        }
      }
    });

    pythonProcess.on("error", (err) => {
      reject(new Error(`Failed to start Python script: ${err.message}`));
    });
  });
}
