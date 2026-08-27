import { useState, useRef } from "react";
import { flushSync } from "react-dom";

const API_BASE = "https://lspp-rag-chatbot.onrender.com";

export default function App() {
  const [messages, setMessages] = useState([]); // {role, content}
  const [question, setQuestion] = useState("");
  const [uploadStatus, setUploadStatus] = useState("No PDF uploaded yet.");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const fileInputRef = useRef(null);

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setUploadStatus(`Uploading "${file.name}"...`);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (data.status === "ok") {
        setUploadStatus(`Loaded "${data.filename}". Ask a question below.`);
      } else {
        setUploadStatus(`Upload failed: ${data.detail}`);
      }
    } catch (err) {
      setUploadStatus(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
    }
  }

  async function handleAsk() {
    if (!question.trim() || asking) return;

    const userMessage = { role: "user", content: question };
    const history = messages;

    let assistantIndex;
    setMessages((prev) => {
      assistantIndex = prev.length + 1; // index of the new assistant message
      return [...prev, userMessage, { role: "assistant", content: "" }];
    });
    setQuestion("");
    setAsking(true);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMessage.content, history }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunkText = decoder.decode(value, { stream: true });

        flushSync(() => {
          setMessages((prev) => {
            if (!prev[assistantIndex]) return prev; // safety guard
            const updated = [...prev];
            updated[assistantIndex] = {
              ...updated[assistantIndex],
              content: updated[assistantIndex].content + chunkText,
            };
            return updated;
          });
        });
      }
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        if (updated[assistantIndex]) {
          updated[assistantIndex] = {
            role: "assistant",
            content: `Error: ${err.message}`,
          };
        }
        return updated;
      });
    } finally {
      setAsking(false);
    }
  }
  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAsk();
    }
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(180deg, #e8ede3 0%, #d9e2d1 100%)",
        fontFamily: "'Segoe UI', sans-serif",
        padding: "40px 20px",
      }}
    >
      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: 28 }}>
          <h1
            style={{
              color: "#3d4f38",
              margin: 0,
              fontSize: 26,
              letterSpacing: -0.5,
            }}
          >
            LSPP Playbook Assistant
          </h1>
          <p style={{ color: "#7a8c74", marginTop: 6, fontSize: 14 }}>
            Ask questions about your uploaded document
          </p>
        </div>

        <div
          style={{
            backgroundColor: "#ffffff",
            borderRadius: 20,
            padding: 16,
            marginBottom: 16,
            boxShadow: "0 2px 10px rgba(74, 93, 69, 0.08)",
            display: "flex",
            alignItems: "center",
            gap: 12,
          }}
        >
          <label
            htmlFor="pdf-upload"
            style={{
              backgroundColor: "#5c7a52",
              color: "white",
              padding: "9px 16px",
              borderRadius: 999,
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
              whiteSpace: "nowrap",
            }}
          >
            Upload PDF
          </label>
          <input
            id="pdf-upload"
            type="file"
            accept=".pdf"
            ref={fileInputRef}
            onChange={handleUpload}
            disabled={uploading}
            style={{ display: "none" }}
          />
          <p
            style={{
              color: "#6b7d64",
              fontSize: 13,
              margin: 0,
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {uploadStatus}
          </p>
        </div>

        <div
          style={{
            backgroundColor: "#ffffff",
            borderRadius: 20,
            padding: 20,
            minHeight: 340,
            maxHeight: 520,
            overflowY: "auto",
            marginBottom: 16,
            boxShadow: "0 2px 10px rgba(74, 93, 69, 0.08)",
          }}
        >
          {messages.length === 0 && (
            <div
              style={{
                textAlign: "center",
                color: "#a8b8a0",
                padding: "60px 0",
                fontSize: 14,
              }}
            >
              Upload a PDF and start asking questions
            </div>
          )}
          {messages.map((m, i) => (
            <div
              key={i}
              style={{
                display: "flex",
                justifyContent: m.role === "user" ? "flex-end" : "flex-start",
                marginBottom: 14,
              }}
            >
              {m.role === "assistant" && (
                <div
                  style={{
                    width: 30,
                    height: 30,
                    borderRadius: "50%",
                    backgroundColor: "#dce6d5",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 12,
                    fontWeight: 700,
                    color: "#4a5d45",
                    marginRight: 8,
                    flexShrink: 0,
                  }}
                >
                  A
                </div>
              )}
              <div
                style={{
                  maxWidth: "75%",
                  padding: "10px 15px",
                  borderRadius:
                    m.role === "user"
                      ? "16px 16px 4px 16px"
                      : "16px 16px 16px 4px",
                  backgroundColor: m.role === "user" ? "#5c7a52" : "#f1f4ee",
                  color: m.role === "user" ? "#ffffff" : "#333",
                  whiteSpace: "pre-wrap",
                  fontSize: 14.5,
                  lineHeight: 1.5,
                }}
              >
                {m.content ||
                  (asking && i === messages.length - 1 ? "..." : "")}
              </div>
            </div>
          ))}
        </div>

        <div
          style={{
            display: "flex",
            gap: 10,
            backgroundColor: "#ffffff",
            borderRadius: 999,
            padding: 6,
            boxShadow: "0 2px 10px rgba(74, 93, 69, 0.08)",
          }}
        >
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about the PDF..."
            style={{
              flex: 1,
              border: "none",
              outline: "none",
              padding: "10px 16px",
              fontSize: 14.5,
              borderRadius: 999,
              fontFamily: "inherit",
              backgroundColor: "transparent",
            }}
          />
          <button
            onClick={handleAsk}
            disabled={asking}
            style={{
              backgroundColor: asking ? "#b8c7b0" : "#5c7a52",
              color: "white",
              border: "none",
              borderRadius: 999,
              padding: "0 22px",
              cursor: asking ? "default" : "pointer",
              fontSize: 14,
              fontWeight: 600,
              flexShrink: 0,
            }}
          >
            {asking ? "Sending" : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
