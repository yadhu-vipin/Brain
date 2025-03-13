// Brain/src/app/api/model/route.js
import { NextResponse } from "next/server";
import { HfInference } from "@huggingface/inference";

export async function POST(request) {
  try {
    const inference = new HfInference(process.env.HF_ACCESS_TOKEN);

    const formData = await request.formData();
    const imageFile = formData.get("file");

    if (!imageFile) {
      return NextResponse.json({ error: "No image file provided" }, { status: 400 });
    }

    const imageBuffer = Buffer.from(await imageFile.arrayBuffer());

    const result = await inference.imageClassification({
      model: process.env.MODEL_ID,
      data: imageBuffer,
    });

    const topPrediction = result.sort((a, b) => b.score - a.score)[0];

    return NextResponse.json({
      prediction: topPrediction.label,
      confidence: topPrediction.score,
      allPredictions: result,
    });

  } catch (error) {
    console.error("Error processing image:", error);

    return NextResponse.json(
      { error: "Failed to process the image" },
      { status: 500 }
    );
  }
}

// Handle GET requests with a 405 response
export function GET() {
  return NextResponse.json({ error: "Method not allowed" }, { status: 405 });
}
