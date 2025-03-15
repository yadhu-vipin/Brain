const path = require("path");
const fs = require("fs");
const ort = require("onnxruntime-node");
const sharp = require("sharp"); // For image preprocessing

// Class labels
const CLASSES = ["no_tumor", "glioma", "meningioma", "pituitary"];

// Load and preprocess the image
async function preprocessImage(base64Image) {
  const imageBuffer = Buffer.from(base64Image, "base64");

  // Preprocess image using Sharp
  const resizedImage = await sharp(imageBuffer)
    .resize(224, 224) // Resize to 224x224
    .toFormat("jpeg")
    .raw()
    .toBuffer();

  // Normalize pixel values (same as in model.py)
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const floatArray = new Float32Array(224 * 224 * 3);

  for (let i = 0; i < floatArray.length; i += 3) {
    floatArray[i] = (resizedImage[i] / 255 - mean[0]) / std[0]; // R channel
    floatArray[i + 1] = (resizedImage[i + 1] / 255 - mean[1]) / std[1]; // G channel
    floatArray[i + 2] = (resizedImage[i + 2] / 255 - mean[2]) / std[2]; // B channel
  }

  return Float32Array.from(floatArray);
}

// Predict using ONNX model
async function predict(base64Image) {
  const modelPath = path.join(__dirname, "../../model.onnx");
  console.log("Resolved model path:", modelPath);


if (!fs.existsSync(modelPath)) {
  throw new Error(`Model file not found at ${modelPath}`);
}

  // Load ONNX model
  const session = await ort.InferenceSession.create(modelPath);

  // Preprocess the image
  const inputTensor = await preprocessImage(base64Image);
  const tensor = new ort.Tensor("float32", inputTensor, [1, 3, 224, 224]);

  // Run inference
  const results = await session.run({ input: tensor });
  const output = results.output.data;

  // Find the class with the highest confidence
  const predictedIdx = output.indexOf(Math.max(...output));
  const confidence = Math.max(...output) * 100;

  // Prepare the response
  return {
    prediction: CLASSES[predictedIdx],
    class_id: predictedIdx,
    confidence: confidence.toFixed(2),
    all_probabilities: CLASSES.reduce((acc, label, idx) => {
      acc[label] = (output[idx] * 100).toFixed(2);
      return acc;
    }, {}),
  };
}

exports.handler = async function (event) {
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

    // Run prediction
    const result = await predict(imageBase64);

    return {
      statusCode: 200,
      body: JSON.stringify(result),
    };
  } catch (error) {
    console.error("Error:", error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message }),
    };
  }
};
