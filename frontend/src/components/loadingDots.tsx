import { FileText } from 'lucide-react';

export default function LoadingDots() {
  return (
    <div className="flex items-start gap-2.5">
      <div className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center bg-gradient-to-br from-emerald-100 to-teal-100 border border-emerald-200">
        <FileText className="w-4 h-4 text-emerald-700" />
      </div>
      <div className="rounded-xl px-4 py-2.5 bg-white/90 backdrop-blur-sm border border-gray-100 shadow-sm">
        <div className="flex gap-1">
          <div className="w-2 h-2 bg-emerald-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-emerald-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-2 h-2 bg-emerald-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    </div>
  );
}