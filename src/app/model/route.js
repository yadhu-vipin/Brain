import { NextResponse } from 'next/server';
import { HfInference } from '@huggingface/inference';

export async function POST(request) {
  try {
    // Initialize Hugging Face inference with your access token
    const inference = new HfInference(process.env.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);
    
    // Get the form data from the request
    const formData = await request.formData();
    const imageFile = formData.get('file');
    
    if (!imageFile) {
      return NextResponse.json(
        { error: 'No image file provided' },
        { status: 400 }
      );
    }
    
    // Convert the file to a Buffer
    const imageBuffer = Buffer.from(await imageFile.arrayBuffer());
    
    // Call your Hugging Face model
    const result = await inference.imageClassification({
      model: process.env.MODEL_ID, // Your private brain tumor model ID
      data: imageBuffer,
    });
    
    // Extract the prediction with highest confidence
    const topPrediction = result.sort((a, b) => b.score - a.score)[0];
    
    // Format the response
    return NextResponse.json({
      prediction: topPrediction.label,
      confidence: topPrediction.score,
      allPredictions: result
    });
    
  } catch (error) {
    console.error('Error processing image:', error);
    
    return NextResponse.json(
      { error: 'Failed to process the image' },
      { status: 500 }
    );
  }
}