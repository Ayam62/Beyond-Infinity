"use client";
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { FileText, MessageSquare, Shield, Globe, ArrowRight, CheckCircle, Sparkles } from 'lucide-react';

export default function Homepage() {
  const router = useRouter();
  const [hoveredFeature, setHoveredFeature] = useState<number | null>(null);

  const features = [
    {
      icon: <MessageSquare className="w-6 h-6" />,
      title: "AI-Powered Chat",
      description: "Get instant answers using advanced AI",
      color: "from-emerald-500 to-teal-500"
    },
    {
      icon: <Shield className="w-6 h-6" />,
      title: "Verified Sources",
      description: "Based on official government documents",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: <Globe className="w-6 h-6" />,
      title: "Bilingual Support",
      description: "Ask in Nepali or English",
      color: "from-purple-500 to-pink-500"
    }
  ];

  const documents = [
    { name: "Passport", nameNp: "राहदानी", icon: "🛂", queries: "2,456+" },
    { name: "Citizenship", nameNp: "नागरिकता", icon: "🏛️", queries: "3,234+" }
  ];

  const popularQuestions = [
    { en: "What documents are required for passport application?", np: "राहदानीका लागि कुन कागजातहरू चाहिन्छ?" },
    { en: "How to apply for citizenship certificate?", np: "नागरिकता प्रमाणपत्रको लागि कसरी आवेदन दिने?" },
    { en: "What is the passport application fee?", np: "राहदानी शुल्क कति हो?" },
    { en: "Citizenship by descent requirements", np: "वंशजका आधारमा नागरिकताको आवश्यकता" }
  ];

  const handleQuestionClick = (question: string) => {
    sessionStorage.setItem('initialQuestion', question);
    router.push('/chat');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-emerald-50">
      
      {/* Navigation */}
      <nav className="bg-white/80 backdrop-blur-md border-b border-gray-200/50 sticky top-0 z-50 shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <button 
            onClick={() => router.push('/')}
            className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          >
            <div className="bg-gradient-to-br from-emerald-600 to-teal-600 rounded-lg p-2 shadow-md">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <div>
<h1 className="text-3xl font-black bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent drop-shadow-md tracking-tighter">
  सहज
</h1>          <p className="text-xs text-gray-600">Government Document Assistant</p>
            </div>
          </button>
          <button 
            onClick={() => router.push('/chat')}
            className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white px-5 py-2 rounded-lg font-semibold transition-all shadow-md hover:shadow-lg text-sm"
          >
            Start Chat
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="max-w-6xl mx-auto px-6 pt-16 pb-12 grid md:grid-cols-2 gap-10 items-center">
        <div className="space-y-5">
          <div className="inline-flex items-center gap-2 bg-emerald-100 text-emerald-700 px-3 py-1.5 rounded-full text-sm font-semibold">
            <Sparkles className="w-4 h-4" />
            <span>AI-Powered • Bilingual Support</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 leading-tight">
            Your Guide to{" "}
            <span className="bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent">
              Government Documents
            </span>
          </h2>
          <p className="text-lg text-gray-600 leading-relaxed">
            Get instant answers about Passport and Citizenship procedures. Ask in Nepali or English.
          </p>
          <div className="flex flex-wrap gap-3 pt-3">
            <button 
              onClick={() => router.push('/chat')}
              className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white px-6 py-3 rounded-xl font-semibold transition-all shadow-lg hover:shadow-xl hover:scale-105 flex items-center gap-2"
            >
              Start Chatting
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>

          <div className="flex items-center gap-6 pt-6">
            <div>
              <div className="text-2xl font-bold text-gray-900">10K+</div>
              <div className="text-xs text-gray-600">Questions</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">99%</div>
              <div className="text-xs text-gray-600">Accuracy</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">2</div>
              <div className="text-xs text-gray-600">Languages</div>
            </div>
          </div>
        </div>

        {/* Chat Preview */}
        <div className="relative">
          <div className="bg-white rounded-2xl shadow-2xl border border-gray-200 overflow-hidden">
            <div className="bg-gradient-to-r from-emerald-600 to-teal-600 p-3 flex items-center justify-between">
              <div className="flex items-center gap-2 text-white">
                <MessageSquare className="w-4 h-4" />
                <span className="font-semibold text-sm">सहज Chat</span>
              </div>
              <div className="flex items-center gap-1 bg-white/20 px-2 py-1 rounded-full">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-white text-xs">Online</span>
              </div>
            </div>

            <div className="p-4 space-y-3 h-80 overflow-y-auto bg-gradient-to-b from-gray-50 to-white">
              <div className="flex gap-2">
                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-emerald-100 to-teal-100 flex items-center justify-center flex-shrink-0">
                  <FileText className="w-3.5 h-3.5 text-emerald-700" />
                </div>
                <div className="bg-white rounded-xl p-2.5 shadow-sm border border-gray-100 max-w-xs">
                  <p className="text-sm text-gray-800">नमस्ते! म तपाईंलाई सरकारी कागजातहरूको बारेमा मद्दत गर्न सक्छु।</p>
                </div>
              </div>

              <div className="flex gap-2 justify-end">
                <div className="bg-gradient-to-r from-emerald-600 to-teal-600 rounded-xl p-2.5 shadow-sm max-w-xs">
                  <p className="text-sm text-white">राहदानीका लागि के कागजात चाहिन्छ?</p>
                </div>
                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center flex-shrink-0">
                  <span className="text-xs">👤</span>
                </div>
              </div>

              <div className="flex gap-2">
                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-emerald-100 to-teal-100 flex items-center justify-center flex-shrink-0">
                  <FileText className="w-3.5 h-3.5 text-emerald-700" />
                </div>
                <div className="bg-white rounded-xl p-2.5 shadow-sm border border-gray-100 max-w-sm">
                  <p className="text-sm text-gray-800 mb-1.5">राहदानीको लागि आवश्यक कागजातहरू:</p>
                  <ul className="text-xs text-gray-700 space-y-0.5 list-disc list-inside">
                    <li>नागरिकता प्रमाणपत्र</li>
                    <li>राहदानी साइजको फोटो</li>
                    <li>आवेदन फारम</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="p-3 bg-gray-50 border-t border-gray-200">
              <div className="flex gap-2">
                <input 
                  type="text" 
                  placeholder="आफ्नो प्रश्न लेख्नुहोस्..."
                  className="flex-1 px-3 py-2 rounded-lg border border-gray-300 text-sm"
                  disabled
                />
                <button className="bg-gradient-to-r from-emerald-600 to-teal-600 text-white p-2 rounded-lg">
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          <div className="absolute -top-3 -right-3 bg-white rounded-lg shadow-lg px-3 py-1.5 border border-emerald-100">
            <div className="flex items-center gap-1.5">
              <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
              <span className="text-xs font-semibold text-gray-900">99% Accurate</span>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="max-w-6xl mx-auto px-6 py-12 grid md:grid-cols-3 gap-4">
        {features.map((feature, idx) => (
          <div
            key={idx}
            onMouseEnter={() => setHoveredFeature(idx)}
            onMouseLeave={() => setHoveredFeature(null)}
            className={`bg-white rounded-xl p-5 border-2 transition-all duration-300 cursor-pointer ${
              hoveredFeature === idx ? 'border-emerald-300 shadow-lg scale-105' : 'border-gray-200 shadow-sm'
            }`}
          >
            <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center text-white mb-3`}>
              {feature.icon}
            </div>
            <h4 className="text-base font-bold text-gray-900 mb-1">{feature.title}</h4>
            <p className="text-gray-600 text-sm">{feature.description}</p>
          </div>
        ))}
      </section>

      {/* Documents */}
      <section className="max-w-6xl mx-auto px-6 py-12">
        <div className="text-center mb-8">
          <h3 className="text-3xl font-bold text-gray-900 mb-2">Supported Documents</h3>
          <p className="text-gray-600">We provide guidance for these government documents</p>
        </div>

        <div className="grid md:grid-cols-2 gap-4 max-w-2xl mx-auto">
          {documents.map((doc, idx) => (
            <div
              key={idx}
              className="bg-white rounded-xl p-6 text-center border-2 border-gray-200 hover:border-emerald-300 hover:shadow-lg transition-all cursor-pointer group"
            >
              <div className="text-5xl mb-3 group-hover:scale-110 transition-transform">{doc.icon}</div>
              <h5 className="font-bold text-gray-900 text-lg mb-1">{doc.name}</h5>
              <p className="text-sm text-gray-600 mb-2">{doc.nameNp}</p>
              <p className="text-xs text-emerald-600 font-semibold">{doc.queries} queries answered</p>
            </div>
          ))}
        </div>
      </section>

      {/* Popular Questions */}
      <section className="max-w-6xl mx-auto px-6 py-12">
        <div className="text-center mb-8">
          <h3 className="text-3xl font-bold text-gray-900 mb-2">Popular Questions</h3>
          <p className="text-gray-600">See what others are asking</p>
        </div>

        <div className="max-w-3xl mx-auto space-y-2.5">
          {popularQuestions.map((question, idx) => (
            <button
              key={idx}
              onClick={() => handleQuestionClick(question.np)}
              className="w-full bg-white hover:bg-emerald-50 rounded-xl p-4 border-2 border-gray-200 hover:border-emerald-300 transition-all text-left group"
            >
              <div className="flex flex-col gap-1">
                <span className="text-gray-900 font-medium text-sm">{question.en}</span>
                <span className="text-gray-500 text-xs">{question.np}</span>
              </div>
            </button>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="max-w-6xl mx-auto px-6 py-12">
        <div className="bg-gradient-to-r from-emerald-600 to-teal-600 rounded-2xl p-10 text-center shadow-xl">
          <h3 className="text-3xl font-bold text-white mb-3">Ready to Get Started?</h3>
          <p className="text-lg text-emerald-50 mb-6">
            Start now and get easy guidance on government documents
          </p>
          <button 
            onClick={() => router.push('/chat')}
            className="bg-white hover:bg-gray-100 text-emerald-600 px-6 py-3 rounded-xl font-bold transition-all shadow-lg hover:shadow-xl hover:scale-105 inline-flex items-center gap-2"
          >
            Start Chatting
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-6xl mx-auto px-6 py-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="bg-gradient-to-br from-emerald-600 to-teal-600 rounded-lg p-1.5">
              <FileText className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-gray-900">सहज</span>
          </div>
          
          <div className="flex gap-6 text-sm text-gray-600">
            <a href="#" className="hover:text-emerald-600">About</a>
            <a href="#" className="hover:text-emerald-600">Privacy</a>
            <a href="#" className="hover:text-emerald-600">Terms</a>
            <a href="#" className="hover:text-emerald-600">Contact</a>
          </div>
          
          <p className="text-xs text-gray-500">© 2025 सहज. Made with ❤️ for Nepal</p>
        </div>
      </footer>
    </div>
  );
}