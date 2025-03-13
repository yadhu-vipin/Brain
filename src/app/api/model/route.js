import { NextResponse } from "next/server";
import { HfInference } from "@huggingface/inference";

export async function POST(request) {
  console.log("✅ API route hit!");

  try {
    console.log("⏳ Extracting form data...");

    const inference = new HfInference(process.env.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);

    // Get the form data from the request
    const formData = await request.formData();
    console.log("📸 Form Data:", formData);

    const imageFile = formData.get("file");
    console.log("📁 Image file received:", imageFile);

    if (!imageFile) {
      console.error("❌ No image file provided");
      return NextResponse.json({ error: "No image file provided" }, { status: 400 });
    }

    // Convert the file to a Buffer
    const imageBuffer = Buffer.from(await imageFile.arrayBuffer());
    console.log("🔍 Image buffer created!");

    // Call Hugging Face model
    console.log("🚀 Sending image to Hugging Face model:", process.env.MODEL_ID);

    const result = await inference.imageClassification({
      model: process.env.MODEL_ID,
      data: imageBuffer,
    });

    console.log("🎯 Hugging Face response:", result);

    // Extract the highest confidence prediction
    const topPrediction = result.sort((a, b) => b.score - a.score)[0];
    console.log("✅ Top prediction:", topPrediction);

    // Return the response
    return NextResponse.json({
      prediction: topPrediction.label,
      confidence: topPrediction.score,
      allPredictions: result,
    });

  } catch (error) {
    console.error("💥 Error processing image:", error);

    return NextResponse.json(
      { error: "Failed to process the image", details: error.message },
      { status: 500 }
    );
  }
}

// Handle GET requests properly (avoid 405 errors)
export async function GET() {
  console.log("❌ GET request blocked");
  return NextResponse.json({ error: "GET method not supported" }, { status: 405 });
}
