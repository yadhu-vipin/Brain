"use client";

import { motion } from "framer-motion";
import Image from "next/image";
import { useState } from "react";

const teamMembers = [
  {
    name: "Ayyappadas M T",
    role: "Lead Developer",
    image: "https://media.licdn.com/dms/image/v2/D4E03AQH7aOdy-2IAww/profile-displayphoto-shrink_400_400/B4EZTa2fdPHcAk-/0/1738838506138?e=1747872000&v=beta&t=t6TM2p8BRruMXgT7UtqcyN4ti2BHrWwmKZKAQmXWFJE",
    bio: "AI specialist with expertise in machine learning and medical image processing. Passionate about using technology to solve healthcare challenges.",
    skills: ["Machine Learning", "Python", "TensorFlow", "Medical Imaging"],
    linkedin: "https://www.linkedin.com/in/ayyappadas-mt-351b252b5/overlay/about-this-profile/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3BdjqV%2FgHCRwm%2FGed%2FTRicEw%3D%3D"
  },{
    name: "Vighnesh H",
    role: "AI/ML",
    image: "https://media.licdn.com/dms/image/v2/D5603AQEgzmJJN2_XVA/profile-displayphoto-shrink_800_800/B56ZTazewbHsAk-/0/1738837718302?e=1747267200&v=beta&t=yhXU_rDrLAWYSGf1U0zjhEm6kOz1iC364Ba3kZ_-7pQ",
    bio: "AI specialist with expertise in machine learning and medical image processing. Passionate about using technology to solve healthcare challenges.",
    skills: ["Machine Learning", "Python", "TensorFlow", "Medical Imaging"],
    linkedin: "https://www.linkedin.com/in/ayyappadas-mt-351b252b5/overlay/about-this-profile/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3BdjqV%2FgHCRwm%2FGed%2FTRicEw%3D%3D"
  },
  {
    name: "Yadhu Vipin",
    role: "UI/UX Designer",
    image: "https://avatars.githubusercontent.com/u/159510686?s=400&u=981a961d2b799e59889ba686ef0f8462c5456850&v=4",
    bio: "Creative designer focused on building intuitive interfaces for healthcare professionals. Skilled in creating accessible and user-friendly medical applications.",
    skills: ["UI/UX Design", "Next JS", "React", "Accessible Design"],
    linkedin: "https://www.linkedin.com/in/yadhu-vipin-104369275/overlay/about-this-profile/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base%3Bxj7epHxiQMu1DHSnUWzHuA%3D%3D"
  },
];

// Animation variants
const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 }
};

export default function AboutUs() {
  const [activeTab, setActiveTab] = useState("story");

  return (
    <div id="About" className="min-h-screen bg-gradient-to-b from-black via-gray-900 to-black text-white flex flex-col items-center px-6 py-24 font-sans">
      {/* About Tabs Section */}
      <motion.div 
        className="w-full max-w-6xl mx-auto mb-24"
        variants={fadeIn}
        initial="hidden"
        animate="visible"
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <div className="flex justify-center mb-8">
          <div className="inline-flex bg-gray-800 rounded-full p-1">
            <button 
              onClick={() => setActiveTab("story")}
              className={`px-6 py-2 rounded-full text-sm font-medium transition-all ${activeTab === "story" ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white"}`}
            >
              Our Story
            </button>
            <button 
              onClick={() => setActiveTab("mission")}
              className={`px-6 py-2 rounded-full text-sm font-medium transition-all ${activeTab === "mission" ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white"}`}
            >
              Mission
            </button>
            <button 
              onClick={() => setActiveTab("technology")}
              className={`px-6 py-2 rounded-full text-sm font-medium transition-all ${activeTab === "technology" ? "bg-blue-600 text-white" : "text-gray-400 hover:text-white"}`}
            >
              Technology
            </button>
          </div>
        </div>
        
        <div className="bg-gray-800 bg-opacity-50 rounded-3xl p-8 md:p-12 border border-gray-700">
          {activeTab === "story" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl font-bold mb-6 text-blue-400">Our Story</h2>
              <p className="text-gray-300 mb-6 leading-relaxed">
                Our journey began when we recognized a critical gap in early brain tumor detection technology. Traditional methods were time-consuming and often required specialized expertise that wasn&apos;t always available, particularly in underserved communities.
              </p>
              <p className="text-gray-300 leading-relaxed">
                By combining our expertise in AI, medical imaging, and user experience design, we&apos;ve created a solution that empowers healthcare providers to detect brain tumors earlier , and based on which they can create better outcomes of treatment .
              </p>
            </motion.div>
          )}
          
          {activeTab === "mission" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl font-bold mb-6 text-purple-400">Our Mission</h2>
              <p className="text-gray-300 mb-6 leading-relaxed">
                We&apos;re on a mission to democratize access to advanced medical diagnostics, making early detection available to all communities regardless of resources or location. We believe that cutting-edge healthcare technology should be accessible to everyone.
              </p>
              <p className="text-gray-300 leading-relaxed">
                Our goal is to reduce diagnosis time by 50% while improving accuracy to over 99%, allowing medical professionals to focus on what matters most: patient care and treatment planning.
              </p>
            </motion.div>
          )}
          
          {activeTab === "technology" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-3xl font-bold mb-6 text-teal-400">Our Technology</h2>
              <p className="text-gray-300 mb-6 leading-relaxed">
                Our AI solution uses a proprietary deep learning algorithm trained on over 100,000 anonymized medical images. The system can detect anomalies that might be missed by the human eye and categorize findings based on urgency and type.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
                <div className="bg-gray-900 p-6 rounded-xl border border-gray-700">
                  <div className="w-12 h-12 bg-blue-900 bg-opacity-30 rounded-lg flex items-center justify-center mb-4">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Deep Learning</h3>
                  <p className="text-gray-400">Advanced neural networks that learn from each scan to continuously improve accuracy.</p>
                </div>
                <div className="bg-gray-900 p-6 rounded-xl border border-gray-700">
                  <div className="w-12 h-12 bg-purple-900 bg-opacity-30 rounded-lg flex items-center justify-center mb-4">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Real-time Analysis</h3>
                  <p className="text-gray-400">Results delivered in seconds, not days, enabling faster medical decisions.</p>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </motion.div>

      {/* Meet the Team Section */}
      <motion.h2
        className="text-4xl font-bold text-center mb-2"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        Meet the <span className="text-blue-400">Team</span>
      </motion.h2>
      
      <motion.p
        className="text-lg text-gray-400 text-center max-w-2xl mb-16"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.6 }}
      >
        Our dedicated team combines expertise in AI, medical imaging, and software development to create innovative solutions for healthcare professionals.
      </motion.p>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-12 w-full max-w-6xl mx-auto">
        {teamMembers.map((member, index) => (
          <motion.div
            key={index}
            className="relative group"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.3, duration: 0.8 }}
          >
            <div className="absolute -inset-px bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl blur opacity-75 group-hover:opacity-100 transition duration-1000 group-hover:duration-200"></div>
            <div className="relative bg-gray-900 p-8 rounded-2xl h-full flex flex-col">
              <div className="flex items-center mb-6">
                <div className="relative w-24 h-24 mr-6">
                  <Image
                    src={member.image}
                    alt={member.name}
                    width={96}
                    height={96}
                    className="rounded-xl object-cover"
                    unoptimized={true} //  Disable Image Optimization
                  />
                  <div className="absolute -bottom-2 -right-2 w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path>
                      <rect x="2" y="9" width="4" height="12"></rect>
                      <circle cx="4" cy="4" r="2"></circle>
                    </svg>
                  </div>
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-white">{member.name}</h3>
                  <p className="text-blue-400 font-medium">{member.role}</p>
                </div>
              </div>
              
              <p className="text-gray-300 mb-6">
                {member.bio}
              </p>
              
              <div className="mt-auto">
                <h4 className="text-sm uppercase tracking-wider text-gray-500 mb-3">Expertise</h4>
                <div className="flex flex-wrap gap-2">
                  {member.skills.map((skill, skillIndex) => (
                    <span 
                      key={skillIndex} 
                      className="text-xs bg-gray-800 text-gray-300 px-3 py-1 rounded-full"
                    >
                      {skill}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
      
    </div>
  );
}
