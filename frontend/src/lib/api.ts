const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function sendMessage(message: string) {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ message })
  });

  if (!response.ok) {
    throw new Error("Failed to send message");
  }

  return response.json();
}

export async function clearChatHistory() {
  const response = await fetch(`${API_BASE_URL}/clear-history`, {
    method: "POST"
  });

  if (!response.ok) {
    throw new Error("Failed to clear chat history");
  }

  return response.json();
}
