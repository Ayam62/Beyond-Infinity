export interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  sources: Source[];
}

export interface Source {
  source_link: string;
  source_type: string;
}

export interface ChatResponse {
  reply: string;
  sources: Source[];
}
