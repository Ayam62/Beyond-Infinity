"use client";
import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Send, FileText, User, CheckCircle2, ExternalLink, Trash2 } from 'lucide-react';
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import LoadingDots from '@/components/loadingDots';

export default function ChatPage() {
  const router = useRouter();
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "नमस्ते! **सहज** मा स्वागत छ। म तपाईंलाई सरकारी कागजात प्रक्रियाहरूमा मद्दत गर्न यहाँ छु। तपाईं नेपाली वा अंग्रेजीमा प्रश्न सोध्न सक्नुहुन्छ।\n\nWelcome! I'm here to help you with government document procedures. You can ask in Nepali or English.",
      sender: 'bot',
      timestamp: new Date(),
      sources: []
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Check for initial question from homepage
  useEffect(() => {
    const initialQuestion = sessionStorage.getItem('initialQuestion');
    if (initialQuestion) {
      sessionStorage.removeItem('initialQuestion');
      setInputValue(initialQuestion);
      // Auto-send after a brief delay
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

  const handleSendMessage = async (messageText?: string) => {
    const textToSend = messageText || inputValue.trim();
    if (textToSend === '') return;

    const userMessage = {
      id: messages.length + 1,
      text: textToSend.replace(/\r?\n/g, '\n'),
      sender: 'user',
      timestamp: new Date(),
      sources: []
    };

    setMessages([...messages, userMessage]);
    setInputValue('');
    setIsLoading(true);

    setTimeout(() => {
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }, 0);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: textToSend })
      });

      if (!response.ok) throw new Error("Network response was not ok");

      const data = await response.json();

      const botMessage = {
        id: messages.length + 2,
        text: data.reply || "No response from the server.",
        sender: 'bot',
        timestamp: new Date(),
        sources: data.sources || []
      };
      console.log(botMessage)
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: messages.length + 2,
        text: "⚠️ Failed to connect to the server. Please ensure the FastAPI backend is running.",
        sender: 'bot',
        timestamp: new Date(),
        sources: []
      };
      setMessages(prev => [...prev, errorMessage]);
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
        await fetch("http://127.0.0.1:8000/clear-history", {
          method: "POST"
        });
        setMessages([
          {
            id: 1,
            text: "नमस्ते! **सहज** मा स्वागत छ। म तपाईंलाई सरकारी कागजात प्रक्रियाहरूमा मद्दत गर्न यहाँ छु। तपाईं नेपाली वा अंग्रेजीमा प्रश्न सोध्न सक्नुहुन्छ।\n\nWelcome! I'm here to help you with government document procedures. You can ask in Nepali or English.",
            sender: 'bot',
            timestamp: new Date(),
            sources: []
          }
        ]);
      } catch (error) {
        console.error("Failed to clear chat history:", error);
      }
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-emerald-50">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-md border-b border-gray-200/50 shadow-sm">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <button 
            onClick={() => router.push('/')}
            className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          >
            <div className="bg-gradient-to-br from-emerald-600 to-teal-600 rounded-lg p-2 shadow-md shadow-emerald-200">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900 tracking-tight">सहज</h1>
              <p className="text-xs text-gray-600 font-medium">सरकारी कागजात सहायक</p>
            </div>
          </button>
          <div className="flex items-center gap-3">
            <button
              onClick={() => router.push('/')}
              className="text-gray-600 hover:text-emerald-600 px-3 py-1.5 rounded-lg hover:bg-emerald-50 transition-colors text-sm font-medium"
            >
              Home
            </button>
            <button
              onClick={handleClearChat}
              className="flex items-center gap-1.5 text-gray-600 hover:text-red-600 px-3 py-1.5 rounded-lg hover:bg-red-50 transition-colors text-sm"
              title="Clear Chat"
            >
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
            <div
              key={message.id}
              className={`flex items-start gap-2.5 ${message.sender === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
            >
              {/* Avatar */}
              <div
                className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center shadow-sm ${
                  message.sender === 'bot'
                    ? 'bg-gradient-to-br from-emerald-100 to-teal-100 border border-emerald-200'
                    : 'bg-gradient-to-br from-slate-100 to-gray-100 border border-gray-200'
                }`}
              >
                {message.sender === 'bot' ? (
                  <FileText className="w-4 h-4 text-emerald-700" />
                ) : (
                  <User className="w-4 h-4 text-gray-700" />
                )}
              </div>

              {/* Message Bubble */}
              <div
                className={`flex flex-col max-w-2xl ${message.sender === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div
                  className={`rounded-xl px-4 py-2.5 shadow-sm ${
                    message.sender === 'bot'
                      ? 'bg-white/90 backdrop-blur-sm text-gray-800 border border-gray-100'
                      : 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white'
                  }`}
                >
                  {/* FIXED: Added prose class for bot messages, inline styles for user messages */}
                  {message.sender === 'bot' ? (
                    <div className="prose prose-slate prose-sm max-w-none">
                      <Markdown 
                        remarkPlugins={[remarkGfm]}
                        components={{
                          h2: ({ node, ...props }) => <h2 className="text-lg font-bold mt-3 mb-2 text-gray-900" {...props} />,
                          h3: ({ node, ...props }) => <h3 className="text-base font-semibold mt-2 mb-1.5 text-gray-800" {...props} />,
                          p: ({ node, ...props }) => <p className="my-1.5 leading-relaxed text-gray-700 text-sm" {...props} />,
                          ul: ({ node, ...props }) => <ul className="list-disc ml-5 my-2 space-y-1 text-sm" {...props} />,
                          ol: ({ node, ...props }) => <ol className="list-decimal ml-5 my-2 space-y-1 text-sm" {...props} />,
                          li: ({ node, ...props }) => <li className="leading-relaxed text-gray-700 text-sm" {...props} />,
                          strong: ({ node, ...props }) => <strong className="font-semibold text-gray-900" {...props} />,
                        }}
                      >
                        {message.text}
                      </Markdown>
                    </div>
                  ) : (
                    <div className="text-white whitespace-pre-wrap text-sm">
                      {message.text}
                    </div>
                  )}

                  {/* Sources */}
                  {message.sender === 'bot' && message.sources && message.sources.length > 0 && (
                    <div className="mt-2.5 pt-2.5 border-t border-gray-200">
                      <div className="text-xs text-gray-600 font-semibold mb-1.5">स्रोतहरू (Sources):</div>
                      <div className="space-y-1">
                        {message.sources.map((source, idx) => (
                          <a
                            key={idx}
                            href={source.source_link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1.5 text-xs text-emerald-700 hover:text-emerald-800 hover:underline transition-colors group"
                          >
                            <ExternalLink className="w-3 h-3 flex-shrink-0 group-hover:scale-110 transition-transform" />
                            <span className="truncate max-w-[400px]" title={source.source_link}>
                              {source.source_type || 'Document'}
                            </span>
                          </a>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                <span className="text-xs text-gray-500 mt-1 px-1 font-medium">
                  {message.timestamp.toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                </span>
              </div>
            </div>
          ))}
          
          {/* Loading Animation */}
          {isLoading && <LoadingDots />}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white/80 backdrop-blur-md border-t border-gray-200/50 shadow-lg">
        <div className="max-w-4xl mx-auto px-4 py-3">
          <div className="relative">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                adjustTextareaHeight();
              }}
              onKeyPress={handleKeyPress}
              placeholder="आफ्नो प्रश्न लेख्नुहोस् / Ask your question..."
              disabled={isLoading}
              className="w-full pl-4 pr-14 py-3 rounded-lg border-2 border-gray-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 resize-none bg-white text-gray-900 placeholder-gray-400 shadow-sm transition-all scrollbar-hide disabled:opacity-50 disabled:cursor-not-allowed text-sm"
              style={{ minHeight: '48px', maxHeight: '160px', height: '48px', overflow: 'hidden' }}
            />
            <button
              onClick={() => handleSendMessage()}
              disabled={inputValue.trim() === '' || isLoading}
              className="absolute bottom-1.5 right-1.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 disabled:from-gray-300 disabled:to-gray-300 disabled:cursor-not-allowed text-white rounded-md p-2 transition-all duration-200 shadow-sm shadow-emerald-200/50 disabled:shadow-none"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2 text-center font-medium">
            Press Enter to send • Shift + Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}