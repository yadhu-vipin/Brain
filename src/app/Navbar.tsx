"use client";

import { motion } from "framer-motion";

export default function Navbar() {
  // Function to scroll to the "brain-tumor-detection" section
  const scrollToUpload = () => {
    const element = document.getElementById("brain-tumor-detection");
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };
  const About = () => {
    const element = document.getElementById("About");
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  const navItems = [
    { name: "Home", action: () => window.scrollTo({ top: 0, behavior: "smooth" }) },
    { name: "Upload", action: scrollToUpload }, // Scrolls to the section
    { name: "About", action: About },
  ];

  return (
    <motion.nav
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6 }}
      className="fixed top-0 left-0 w-full flex justify-between items-center px-8 py-4 bg-black/70 backdrop-blur-sm shadow-xl z-50"
    >
      <h1 className="text-3xl font-bold text-gray-100 tracking-wide">Brainiac</h1>
      <div className="flex space-x-6">
        {navItems.map((item, index) => (
          <button
            key={index}
            onClick={item.action}
            className="px-4 py-2 text-lg text-gray-300 hover:text-white transition-all duration-300 ease-in-out"
          >
            {item.name}
          </button>
        ))}
      </div>
    </motion.nav>
  );
}
