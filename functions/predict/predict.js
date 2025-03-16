const path = require("path");
const ort = require("onnxruntime-node");
const sharp = require("sharp");

// Class labels
const CLASSES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"];

// Load and preprocess the image
async function preprocessImage(base64Image) {
  const imageBuffer = Buffer.from(base64Image, "base64");

  // Preprocess image using Sharp to match PyTorch transforms
  const resizedImage = await sharp(imageBuffer)
    .resize(224, 224) // Resize to 224x224
    .raw()
    .toBuffer();

  // Convert to RGB format and normalize using the same values as PyTorch
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const floatArray = new Float32Array(3 * 224 * 224);
  
  // Convert RGB pixel values to tensor format
  // Note: sharp gives us RGB values in a flat array, we need to reformat to CHW
  for (let y = 0; y < 224; y++) {
    for (let x = 0; x < 224; x++) {
      const pixelIndex = (y * 224 + x) * 3;
      
      // In PyTorch/ONNX format we need CHW instead of HWC
      // R channel
      floatArray[0 * 224 * 224 + y * 224 + x] = 
        (resizedImage[pixelIndex] / 255.0 - mean[0]) / std[0];
      // G channel
      floatArray[1 * 224 * 224 + y * 224 + x] = 
        (resizedImage[pixelIndex + 1] / 255.0 - mean[1]) / std[1];
      // B channel
      floatArray[2 * 224 * 224 + y * 224 + x] = 
        (resizedImage[pixelIndex + 2] / 255.0 - mean[2]) / std[2];
    }
  }

  return floatArray;
}

// Predict using ONNX model
async function predict(base64Image) {
  // Use absolute path to reference model.onnx in the project root
  const modelPath = path.resolve(__dirname, "../../model.onnx");

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