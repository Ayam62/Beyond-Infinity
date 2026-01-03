"use client";
import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Send, FileText, User, CheckCircle2, ExternalLink, Trash2, Mic, MicOff, Loader2, MapPin } from 'lucide-react';
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import LoadingDots from '@/components/loadingDots';
import hark from 'hark';
import dynamic from 'next/dynamic';

// Dynamic import for Leaflet (Required for Next.js SSR)
const OfficeMap = dynamic(() => import('@/components/OfficeMap'), { 
  ssr: false,
  loading: () => <div className="h-48 w-full bg-gray-100 animate-pulse rounded-lg flex items-center justify-center text-xs text-gray-400">Loading Map...</div>
});

export default function ChatPage() {
  const router = useRouter();
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "नमस्ते! **सहज** मा स्वागत छ। म तपाईंलाई सरकारी कागजात प्रक्रियाहरूमा मद्दत गर्न यहाँ छु। तपाईं नेपाली वा अंग्रेजीमा प्रश्न सोध्न सक्नुहुन्छ।\n\nWelcome! I'm here to help you with government document procedures. You can ask in Nepali or English.",
      sender: 'bot',
      timestamp: new Date(),
      sources: [],
      nearest_office: null // Added to initial state
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  
  const [location, setLocation] = useState({ latitude: null, longitude: null });

  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const silenceTimerRef = useRef(null);
  const harkRef = useRef(null);

  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  useEffect(() => {
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLocation({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          });
        },
        (error) => {
          console.error("Error obtaining location:", error);
        }
      );
    }
  }, []);

  useEffect(() => {
    const initialQuestion = sessionStorage.getItem('initialQuestion');
    if (initialQuestion) {
      sessionStorage.removeItem('initialQuestion');
      setInputValue(initialQuestion);
      setTimeout(() => {
        handleSendMessage(initialQuestion);
      }, 500);
    }
  }, []);

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      const speechEvents = hark(stream, { threshold: -50, interval: 100 });
      harkRef.current = speechEvents;

      speechEvents.on('speaking', () => {
        if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
      });

      speechEvents.on('stopped_speaking', () => {
        silenceTimerRef.current = setTimeout(() => {
          if (mediaRecorder.state !== "inactive") {
            stopRecording();
          }
        }, 2000);
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await handleTranscriptionOnly(audioBlob);
        stream.getTracks().forEach(track => track.stop());
        if (harkRef.current) harkRef.current.stop();
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Mic Error:", err);
      alert("Microphone access denied.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    }
  };

  const handleTranscriptionOnly = async (audioBlob) => {
    setIsTranscribing(true);
    const formData = new FormData();
    formData.append("file", audioBlob, "user_speech.webm");

    try {
      const response = await fetch("http://127.0.0.1:8000/transcribe", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Transcription failed");
      const data = await response.json();

      if (data.transcribed_text) {
        setInputValue(data.transcribed_text);
        setTimeout(adjustTextareaHeight, 100);
      }
    } catch (error) {
      console.error("Error transcribing:", error);
    } finally {
      setIsTranscribing(false);
    }
  };

  const handleSendMessage = async (messageText) => {
    const textToSend = messageText || inputValue.trim();
    if (textToSend === '' || isTranscribing) return;

    const userMessage = {
      id: Date.now(),
      text: textToSend.replace(/\r?\n/g, '\n'),
      sender: 'user',
      timestamp: new Date(),
      sources: []
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    setTimeout(() => {
      if (textareaRef.current) textareaRef.current.style.height = '48px';
    }, 0);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: textToSend,
          latitude: location.latitude,
          longitude: location.longitude 
        })
      });
      const data = await response.json();
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: data.reply,
        sender: 'bot',
        timestamp: new Date(),
        sources: data.sources || [],
        nearest_office: data.nearest_office // Captured coordinates
      }]);
    } catch (error) {
      console.error("Chat error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleClearChat = async () => {
    if (confirm('Are you sure you want to clear the chat history?')) {
      try {
        await fetch("http://127.0.0.1:8000/clear-history", { method: "POST" });
        setMessages([messages[0]]);
      } catch (error) { console.error(error); }
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-emerald-50">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-md border-b border-gray-200/50 shadow-sm">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <button onClick={() => router.push('/')} className="flex items-center gap-3 hover:opacity-80 transition-opacity">
            <div className="bg-gradient-to-br from-emerald-600 to-teal-600 rounded-lg p-2 shadow-md shadow-emerald-200">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 tracking-tight">सहज</h1>
              <p className="text-xs text-gray-600 font-medium">सरकारी कागजात सहायक</p>
            </div>
          </button>
          <div className="flex items-center gap-3">
            <button onClick={handleClearChat} className="flex items-center gap-1.5 text-gray-600 hover:text-red-600 px-3 py-1.5 rounded-lg hover:bg-red-50 transition-colors text-sm">
              <Trash2 className="w-4 h-4" />
              <span className="hidden sm:inline">Clear</span>
            </button>
            <div className="flex items-center gap-1.5 text-emerald-700 bg-gradient-to-r from-emerald-50 to-teal-50 px-3 py-1.5 rounded-lg border border-emerald-100">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span className="text-xs font-semibold">Verified</span>
            </div>
          </div>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.map((message) => (
            <div key={message.id} className={`flex items-start gap-2.5 ${message.sender === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
              <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center shadow-sm ${message.sender === 'bot' ? 'bg-gradient-to-br from-emerald-100 to-teal-100 border border-emerald-200' : 'bg-gradient-to-br from-slate-100 to-gray-100 border border-gray-200'}`}>
                {message.sender === 'bot' ? <FileText className="w-4 h-4 text-emerald-700" /> : <User className="w-4 h-4 text-gray-700" />}
              </div>
              <div className={`flex flex-col max-w-2xl ${message.sender === 'user' ? 'items-end' : 'items-start'}`}>
                <div className={`rounded-xl px-4 py-2.5 shadow-sm ${message.sender === 'bot' ? 'bg-white/90 backdrop-blur-sm text-gray-800 border border-gray-100' : 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white'}`}>
                  {message.sender === 'bot' ? (
                    <div className="prose prose-slate prose-sm max-w-none">
                      <Markdown remarkPlugins={[remarkGfm]}>{message.text}</Markdown>
                      
                      {/* Map rendered inside bot message if coordinates exist */}
                      {message.nearest_office && (
                        <div className="mt-4 pt-4 border-t border-gray-100">
                          <div className="flex items-center gap-2 mb-2 text-emerald-700 font-bold text-xs">
                             <MapPin size={14} /> नजिकको कार्यालय (Nearest Office Map)
                          </div>
                          <OfficeMap 
                            lat={message.nearest_office.latitude}
                            lon={message.nearest_office.longitude}
                            name={message.nearest_office.name}
                            address={message.nearest_office.address}
                          />
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-white whitespace-pre-wrap text-sm">{message.text}</div>
                  )}
                  {message.sender === 'bot' && message.sources && message.sources.length > 0 && (
                    <div className="mt-2.5 pt-2.5 border-t border-gray-200">
                        <div className="text-xs text-gray-600 font-semibold mb-1.5">स्रोतहरू (Sources):</div>
                        {message.sources.map((source, idx) => (
                            <a key={idx} href={source.source_link} target="_blank" className="flex items-center gap-1.5 text-xs text-emerald-700 hover:underline">
                                <ExternalLink className="w-3 h-3" /> {source.source_type || 'Document'}
                            </a>
                        ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
          {isLoading && <LoadingDots />}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white/80 backdrop-blur-md border-t border-gray-200/50 shadow-lg">
        <div className="max-w-4xl mx-auto px-4 py-3">
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
                <textarea
                  ref={textareaRef}
                  value={inputValue}
                  onChange={(e) => { setInputValue(e.target.value); adjustTextareaHeight(); }}
                  onKeyPress={handleKeyPress}
                  placeholder={
                    isRecording ? "Listening (बोलिरहनुहोस्)..." : 
                    isTranscribing ? "Typing from voice..." : 
                    "आफ्नो प्रश्न लेख्नुहोस् / Ask your question..."
                  }
                  disabled={isLoading || isRecording || isTranscribing}
                  className="w-full pl-4 pr-12 py-3 rounded-lg border-2 border-gray-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 resize-none bg-white text-gray-900 text-sm disabled:opacity-50"
                  style={{ minHeight: '48px', maxHeight: '160px', height: '48px', overflow: 'hidden' }}
                />
                
                {isTranscribing && (
                  <div className="absolute right-14 bottom-3">
                    <Loader2 className="w-4 h-4 text-emerald-600 animate-spin" />
                  </div>
                )}

                <button
                  onClick={() => handleSendMessage()}
                  disabled={inputValue.trim() === '' || isLoading || isRecording || isTranscribing}
                  className="absolute bottom-1.5 right-1.5 bg-gradient-to-r from-emerald-600 to-teal-600 text-white rounded-md p-2 disabled:opacity-50"
                >
                  <Send className="w-4 h-4" />
                </button>
            </div>

            <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isLoading || isTranscribing}
                className={`p-3 rounded-full transition-all duration-300 ${
                    isRecording 
                    ? 'bg-red-500 text-white animate-pulse' 
                    : 'bg-emerald-100 text-emerald-700 hover:bg-emerald-200 disabled:opacity-30'
                }`}
                title={isRecording ? "Stop Recording" : "Start Voice Input"}
            >
                {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2 text-center font-medium">
            {isRecording ? "Automatic stop after 2s of silence" : "Use the mic to speak in Nepali or English"}
          </p>
        </div>
      </div>
    </div>
  );
}