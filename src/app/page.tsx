"use client";

import Navbar from "../app/Navbar";
import MediaUploadSection from "../app/Holo-upload";
import About from "../app/about";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Navbar */}
      <Navbar />

      {/* Main Content */}
      <main className="flex-1">
        {/* Upload Section */}
        <div  className="flex justify-center items-center min-h-screen">
          <MediaUploadSection />
        </div>

        {/* About Section (Full Width) */}
        <div className="w-full">
          <About />
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-black text-gray-400 text-center p-4">
        <p>© 2024 Brainiac. All rights reserved.</p>
      </footer>
    </div>
  );
}
