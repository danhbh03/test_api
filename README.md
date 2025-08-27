# 📘 README.md

## 1. Giới thiệu
Repo này chứa 2 chatbot dựa trên **LightRAG**:
- `lightRAGChatbot.py`: chatbot RAG cơ bản (tiếng Anh).
- `lightRAGChatbotMultilang.py`: chatbot RAG đa ngôn ngữ (normalize → EN, Jina Embeddings v3, dịch bằng Claude, fallback Google Translate).

## 2. Yêu cầu hệ thống
- Python **3.11.13**
- Máy có thể kết nối internet để gọi API của:
  - [Anthropic (Claude)](https://console.anthropic.com/dashboard)
  - [Jina AI (Embeddings)](https://jina.ai/api-dashboard/embedding)
  - [Groq LLM](https://console.groq.com/home)

## 3. Cài đặt
Clone repo và cài dependency:

```bash
git clone <repo-url>
cd <repo>
python3.11 -m venv venv
source venv/bin/activate   # hoặc venv\Scripts\activate trên Windows

# Cài dependencies cần thiết
pip install -r requirements.txt
```

### Dependencies chính
- `gradio` (UI cho chatbot)  
- `lightrag` (RAG engine)  
- `transformers` + `sentencepiece` (xử lý mô hình NLP)  
- `python-docx` (đọc file DOCX)  
- `regex`, `scikit-learn` (xử lý text & vector)  
- `nest_asyncio` (async runtime)  
- `langdetect` (phát hiện ngôn ngữ)  
- `deep-translator` (Google Translate fallback)  
- `httpx` (HTTP client)  
- `python-dotenv` (quản lý `.env`)  
- `groq` (client cho Groq API)  

## 4. Thiết lập API keys
Tạo file `.env` trong thư mục gốc với nội dung:

```env
# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Jina AI
JINA_API_KEY=your_jina_key_here

# Groq
GROQ_API_KEY=your_groq_key_here

# Claude model (không cần chỉ định nếu muốn mặc định)
# ANTHROPIC_MODEL=claude-3-5-sonnet-20240620
```

> ⚠️ Lưu ý: nếu không có Claude API hoặc lỗi quota, hệ thống sẽ tự động fallback sang **Google Translate** để đảm bảo chatbot hoạt động.

## 5. Chạy chatbot

### 5.1 Chatbot tiếng Anh (cơ bản)
```bash
python lightRAGChatbot.py
```

Truy cập Gradio UI tại:
```
http://127.0.0.1:9621
```

### 5.2 Chatbot đa ngôn ngữ
```bash
python lightRAGChatbotMultilang.py
```

Truy cập Gradio UI tại:
```
http://127.0.0.1:9621
```

Ở đây bạn có thể nhập câu hỏi bằng nhiều ngôn ngữ (tiếng Việt, Nhật, Pháp, …).  
Pipeline sẽ:
1. Detect ngôn ngữ → dịch sang **tiếng Anh** bằng Claude.  
2. Query LightRAG với embedding từ **Jina v3**.  
3. Trả lời bằng tiếng Anh → dịch ngược sang ngôn ngữ gốc.  
4. Nếu Claude lỗi → fallback Google Translate.  

## 6. Thêm tài liệu vào RAG
- Nhấn nút 📥 `Thêm tài liệu` để upload file `.txt` hoặc `.docx`.  
- Văn bản sẽ được dịch sang tiếng Anh (nếu cần) rồi index bằng **Jina embeddings**.  
- Có thể hỏi chatbot bằng nhiều ngôn ngữ.

## 7. Ghi chú
- Các file được index lưu trong thư mục `./chat_rag_storage`.  
- Mapping doc_id ↔ ngôn ngữ gốc lưu ở `doc_lang_map.json`.  
- Model embedding: mặc định **jina-embeddings-v3**, bạn có thể thay bằng local model như **bge-m3** nếu muốn (phần code đã có chỗ hook-in).  
