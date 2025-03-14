"use client";

import { useState, useEffect, useRef } from "react";
import { Upload, Brain, Loader, Info } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import Image from 'next/image';

export default function MediaUploadSection() {
  const defaultVideo = "";
  const [mediaSrc, setMediaSrc] = useState<string>(defaultVideo);
  const [fileName, setFileName] = useState<string>("");
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [result, setResult] = useState<string>("");
  const [prediction, setPrediction] = useState<string>("");
  const [showRing, setShowRing] = useState<boolean>(false);
  const [activeDemoTab] = useState<string>("upload");
  const [isIntroVisible] = useState<boolean>(true);
  const [showResultDetails, setShowResultDetails] = useState<boolean>(false);
  const ringRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setShowRing(true);
  }, []);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);

      if (mediaSrc !== defaultVideo) URL.revokeObjectURL(mediaSrc);

      setMediaSrc(url);
      setFileName(file.name);
      setResult("");
      setPrediction("");
      setShowResultDetails(false);
    }
  };

  const handleScan = async () => {
    setIsScanning(true);
    setResult("Analyzing image...");

    try {
      const file = (document.querySelector('input[type="file"]') as HTMLInputElement)?.files?.[0];
      const reader = new FileReader();

      reader.onloadend = async () => {
        const base64String = reader.result?.toString().split(",")[1];

        const response = await fetch("/.netlify/functions/predict", {
          method: "POST",
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: base64String }),
        });

        const data = await response.json();

        console.log("Backend response:", data);


        if (response.ok) {
          setResult("Analysis complete!");
          setPrediction(data.prediction);
        } else {
          setResult(data.error || "Analysis failed. Please try again.");
        }
      };

      if (file) {
        reader.readAsDataURL(file);
      }
    } catch (error) {
      setResult("Connection error. Please check your network.");
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <div className="w-screen bg-gradient-to-b from-black via-gray-1200 to-black text-white flex flex-col items-center font-sans">
      <AnimatePresence>
        {isIntroVisible && (
          <motion.div
            className="h-screen w-full flex flex-col items-center justify-center relative overflow-hidden"
            initial={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { duration: 1.9 } }}
          >
            {showRing && (
              <motion.div
                ref={ringRef}
                className={`
      absolute bottom-[260px] w-[400px] h-[40px]
      bg-transparent 
      border-[30px] border-transparent  
      shadow-[0_0_120px_40px_rgba(99,102,241,0.5)] 
      rounded-[60%] 
      transform perspective-[1200px] rotate-x-[60deg] skew-x-2 
      clip-path-[inset(50%_0px_0px_0px)] 
      animate-pulse-glow
      z-10
    `}
                initial={{
                  width: 0,
                  height: 0,
                  boxShadow: "0 0 0 0 rgba(99,102,241,0)",
                  borderWidth: "0px"
                }}
                animate={{
                  width: "400px",
                  height: "40px",
                  boxShadow: "0 0 120px 40px rgba(99,102,241,0.5)",
                  borderWidth: "30px"
                }}
                transition={{
                  duration: 2,
                  ease: "easeOut",
                  delay: 0.3
                }}
              ></motion.div>
            )}

            <motion.div
              className="relative w-[550px] h-[300px] rounded-lg overflow-hidden z-20"
              initial={{ y: -1000, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ duration: 2, ease: "easeOut" }}
            >
              <div className="absolute inset-0 z-10"></div>
              <video
                className="w-full h-full object-cover"
                autoPlay loop muted playsInline
              >
                <source src={mediaSrc} type="video/mp4" />
              </video>
            </motion.div>

            <motion.div
              className="absolute center left-10 w-full p-4 z-20"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 2, duration: 1 }}
            >
              <h2 className="text-7xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text">
                Brainiac AI
              </h2>
              <p className="text-3xl text-gray-300 mt-2">Revolutionary neural tumor detection</p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <div id="brain-tumor-detection" className="flex flex-col items-center justify-center w-full min-h-screen p-6 md:p-12 bg-gray-900">
        <motion.div
          className="w-full max-w-5xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="bg-gray-800 rounded-xl shadow-2xl overflow-hidden border border-gray-700">
            <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-6 border-b border-gray-700">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                  <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text">
                    Brain Tumor Detection
                  </h1>
                  <p className="text-gray-400 mt-2">
                    Advanced AI-powered neural analysis for medical professionals
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-0">
              <div className="lg:col-span-1 p-6 border-r border-gray-700 bg-gray-850">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold text-gray-200">Image Input</h2>
                </div>

                {activeDemoTab === "upload" ? (
                  <>
                    <div className="aspect-square rounded-lg overflow-hidden bg-gray-900 border border-gray-700 mb-4">
                      {mediaSrc.endsWith('.mp4') ? (
                        <video className="w-full h-full object-cover" autoPlay loop muted playsInline>
                          <source src={mediaSrc} type="video/mp4" />
                        </video>
                      ) : (
                        <div className="w-full h-full flex items-center justify-center bg-gray-900">
                          {mediaSrc !== defaultVideo ? (
                            <Image
                              src={mediaSrc}
                              alt="MRI Scan"
                              width={500}
                              height={500}
                              className="max-w-full max-h-full object-contain"
                              unoptimized={true}
                            />
                          ) : (
                            <div className="text-gray-500 text-center p-4">
                              <Upload className="w-12 h-12 mx-auto mb-2" />
                              <p>No image selected</p>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    <label className="flex items-center justify-center w-full bg-gray-700 hover:bg-gray-600 text-white font-medium py-3 px-6 rounded-lg cursor-pointer transition-all shadow-lg border border-gray-600 mb-4">
                      <Upload className="mr-2 w-5 h-5 text-blue-400" />
                      <span>Choose MRI Image</span>
                      <input type="file" accept="image/*" className="hidden" onChange={handleFileUpload} />
                    </label>

                    {fileName && (
                      <div className="flex items-center bg-gray-900 rounded p-3 mb-4">
                        <div className="w-8 h-8 rounded bg-blue-500 bg-opacity-20 flex items-center justify-center mr-3">
                          <Info className="w-4 h-4 text-blue-400" />
                        </div>
                        <div className="overflow-hidden">
                          <p className="text-gray-300 truncate text-sm">📄 {fileName}</p>
                        </div>
                      </div>
                    )}
                  </>
                ) : null}

                <button
                  onClick={() => {
                    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
                    if (fileInput?.files?.[0]) {
                      handleScan();
                    }
                  }}
                  className="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-3 px-6 rounded-lg flex items-center justify-center transition-all shadow-lg"
                  disabled={isScanning || !fileName}
                >
                  {isScanning ? (
                    <>
                      <Loader className="mr-2 w-5 h-5 animate-spin" />
                      <span>Scanning...</span>
                    </>
                  ) : (
                    <>
                      <Brain className="mr-2 w-5 h-5" />
                      <span>Start Scan</span>
                    </>
                  )}
                </button>
              </div>

              <div className="lg:col-span-1 p-6 border-r border-gray-700 bg-gray-800">
                <h2 className="text-xl font-semibold text-gray-200 mb-4">Analysis Result</h2>

                <div className="mb-6">
                  {result ? (
                    <div className="bg-gray-700 rounded-lg p-4">
                      <p className="text-gray-300">{result}</p>
                      {prediction && (
                        <div className="mt-4">
                          <h3 className="text-lg font-semibold text-gray-200">Detected Tumor Type:</h3>
                          <p className="text-sm text-gray-400">{prediction}</p>
                          <button
                            onClick={() => setShowResultDetails(true)}
                            className="mt-3 text-blue-400 hover:text-blue-300 text-sm underline"
                          >
                            View Details
                          </button>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-gray-500 text-center p-4 bg-gray-900 rounded-lg border border-gray-700">
                      <Info className="w-12 h-12 mx-auto mb-2" />
                      <p>No results yet. Upload an image and start the scan.</p>
                    </div>
                  )}
                </div>

                {showResultDetails && prediction && (
                  <motion.div
                    className="bg-gray-800 rounded-lg p-4 border border-gray-700"
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                  >
                    <h3 className="text-lg font-semibold text-gray-200 mb-3">Tumor Type</h3>
                    <p className="text-gray-400 mb-2">
                      <span className="font-medium">{prediction}</span>
                    </p>
                    <button
                      onClick={() => setShowResultDetails(false)}
                      className="mt-4 text-blue-400 hover:text-blue-300 text-sm underline"
                    >
                      Hide Details
                    </button>
                  </motion.div>
                )}
              </div>

            </div>
          </div>

          {/* Key Benefits Section */}
          <div className="mt-8 p-6 bg-gray-800 rounded-xl border border-gray-700">
            <h2 className="text-xl font-semibold text-gray-200 mb-4">Key Benefits</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gradient-to-br from-blue-900 to-blue-950 p-4 rounded-lg border border-blue-800">
                <div className="w-10 h-10 rounded-full bg-blue-500 bg-opacity-20 flex items-center justify-center mb-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="font-medium text-white mb-2">Real-time Analysis</h3>
                <p className="text-sm text-gray-300">Rapid detection capabilities reduce diagnosis time from days to seconds.</p>
              </div>

              <div className="bg-gradient-to-br from-purple-900 to-purple-950 p-4 rounded-lg border border-purple-800">
                <div className="w-10 h-10 rounded-full bg-purple-500 bg-opacity-20 flex items-center justify-center mb-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 4.75 7.5 4.75a4.5 4.5 0 00-4.5 4.5c0 1.789 0.896 3.221 2.247 4.036A4.495 4.495 0 0012 9.75v7.747c0 1.169.883 2.105 2 2.105a4.5 4.5 0 004.5-4.5c0-1.789-.896-3.221-2.247-4.036A4.495 4.495 0 0012 14.747V6.253z" />
                  </svg>
                </div>
                <h3 className="font-medium text-white mb-2">Enhanced Accuracy</h3>
                <p className="text-sm text-gray-300">Achieve up to 98% accuracy in detecting even the smallest tumors.</p>
              </div>

              <div className="bg-gradient-to-br from-green-900 to-green-950 p-4 rounded-lg border border-green-800">
                <div className="w-10 h-10 rounded-full bg-green-500 bg-opacity-20 flex items-center justify-center mb-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <h3 className="font-medium text-white mb-2">Non-invasive Procedure</h3>
                <p className="text-sm text-gray-300">Analyzes existing MRI scans, eliminating the need for invasive biopsies.</p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}