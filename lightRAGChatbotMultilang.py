import os
import json
import asyncio
import time
import re
import unicodedata
from collections import deque
from dotenv import load_dotenv
from collections import deque
import time

import gradio as gr
from docx import Document

from langdetect import detect
from deep_translator import GoogleTranslator

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.kg.shared_storage import initialize_pipeline_status

from groq import Groq
import httpx

# ========= ENV & SETUP =========
load_dotenv()
JINA_API_KEY = os.getenv("JINA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
USE_CLAUDE_TRANSLATION = True  # bật dịch bằng Claude

if JINA_API_KEY is None:
    raise RuntimeError("Please set JINA_API_KEY in your environment (.env).")
if GROQ_API_KEY is None:
    raise RuntimeError("Please set GROQ_API_KEY in your environment (.env).")
if USE_CLAUDE_TRANSLATION and ANTHROPIC_API_KEY is None:
    raise RuntimeError("Please set ANTHROPIC_API_KEY in your environment (.env).")

setup_logger("lightrag", level="INFO")

WORKING_DIR = os.path.abspath("./chat_rag_storage")
os.makedirs(WORKING_DIR, exist_ok=True)
os.environ["LIGHTRAG_WORKING_DIR"] = WORKING_DIR

print("Working dir:", WORKING_DIR)
print("Initial files:", os.listdir(WORKING_DIR))

# ========= CLIENTS =========
client = Groq(api_key=GROQ_API_KEY)

# ========= GROQ LLM (LightRAG) =========
async def groq_model_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] | None = None,
    **kwargs
) -> str:
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=kwargs.get("model", "llama-3.3-70b-versatile"),
            messages=messages,
            temperature=kwargs.get("temperature", 0.3),
            top_p=kwargs.get("top_p", 1),
            max_tokens=kwargs.get("max_tokens", 1024),
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"[ERROR from Groq LLM] {e}") from e

# ========= JINA v3 EMBEDDINGS =========
JINA_RPM = 90  # chỉnh theo quota tài khoản của bạn
_jina_request_times = deque()
async def _jina_rate_gate():
    now = time.monotonic()


    while _jina_request_times and now - _jina_request_times[0] > 60:
        _jina_request_times.popleft()


    if len(_jina_request_times) >= JINA_RPM:
        sleep_time = 60 - (now - _jina_request_times[0])
        await asyncio.sleep(max(0.5, sleep_time))

    _jina_request_times.append(time.monotonic())

async def jina_embed(texts: list[str]) -> list[list[float]]:
    await _jina_rate_gate()   # ⭐ FIX RPM

    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    payload = {
        "model": "jina-embeddings-v3",
        "task": "text-matching",
        "truncate": True,
        "input": texts if isinstance(texts, list) else [texts]
    }

    for attempt in range(3):  # ⭐ retry khi 429
        try:
            async with httpx.AsyncClient(timeout=30) as ac:
                resp = await ac.post(url, headers=headers, json=payload)

            if resp.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue

            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]

        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(1.5)


# ========= TIỀN XỬ LÝ =========
def read_file_text(file):
    if file is None:
        return ""
    if hasattr(file, "name") and file.name.endswith(".docx"):
        doc = Document(file.name)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    with open(file.name, "r", encoding="utf-8") as f:
        return f.read()

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\t", " ")
    s = re.sub(r"[ ]{2,}", " ", s)
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def sentence_split(text: str):
    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def pre_split(text: str, max_block_tokens: int = 2000, tokenizer=None):
    sentences = sentence_split(text)
    blocks, cur, cur_len = [], [], 0
    for s in sentences:
        s_len = len(s.split())
        if cur_len + s_len > max_block_tokens and cur:
            blocks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += s_len
    if cur:
        blocks.append(" ".join(cur))
    return blocks

def enforce_max_len(chunk: str, max_words: int = 300, overlap: int = 50):
    words = chunk.split()
    if len(words) <= max_words:
        return [chunk]
    out, step = [], max_words - max(0, overlap)
    for i in range(0, len(words), step):
        sub = words[i:i+max_words]
        out.append(" ".join(sub))
    return out

def split_paragraph_semantic(paragraph: str, max_words: int = 300, overlap: int = 50):
    blocks = pre_split(paragraph, max_block_tokens=1200)
    chunks = []
    for b in blocks:
        chunks.extend(enforce_max_len(b, max_words=max_words, overlap=overlap))
    return chunks

# ========= DETECT & TRANSLATE HELPERS =========
def detect_lang(text: str, default="en"):
    try:
        return detect(text)
    except Exception:
        return default
ANTHROPIC_RPM = 5
ANTHROPIC_IPM = 10_000
ANTHROPIC_OPM = 4_000

_request_timestamps = deque()
_ipm_used = 0
_opm_used = 0
_window_start = time.monotonic()

def _estimate_tokens(s: str) -> int:
    return max(1, len(s) // 4)

async def _anthropic_rate_gate(tokens_in: int, tokens_out_budget: int):
    global _request_timestamps, _ipm_used, _opm_used, _window_start
    now = time.monotonic()
    if now - _window_start >= 60:
        _request_timestamps.clear()
        _ipm_used = 0
        _opm_used = 0
        _window_start = now

    while True:
        reqs_last_min = len(_request_timestamps)
        if (reqs_last_min < ANTHROPIC_RPM and
            _ipm_used + tokens_in <= ANTHROPIC_IPM and
            _opm_used + tokens_out_budget <= ANTHROPIC_OPM):
            break
        sleep_s = max(0.0, 60 - (time.monotonic() - _window_start))
        await asyncio.sleep(sleep_s)
        now = time.monotonic()
        _request_timestamps.clear()
        _ipm_used = 0
        _opm_used = 0
        _window_start = now

    _request_timestamps.append(now)
    _ipm_used += tokens_in
    _opm_used += tokens_out_budget

import re
import httpx

# ==========================
# Clean translation
# ==========================
def clean_translation(text: str) -> str:
    """
    Extract translated text from <OUTPUT>...</OUTPUT> block
    and clean up common formatting issues.
    """
    if not text:
        return text.strip()

    # 1. Lấy nội dung trong <OUTPUT>...</OUTPUT>
    match = re.search(r"<OUTPUT>\s*(.*?)\s*</OUTPUT>", text, flags=re.S)
    if match:
        text = match.group(1)

    # 2. Trim whitespace
    text = text.strip()

    # 3. Bỏ ngoặc kép/ngoặc đơn nếu bao toàn bộ
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")) or \
       (text.startswith("“") and text.endswith("”")) or \
       (text.startswith("(") and text.endswith(")")):
        text = text[1:-1].strip()

    return text


# ==========================
# Call Claude for translation
# ==========================
async def _claude_translate_chunk(text: str, src_lang: str, tgt_lang: str, max_output_tokens: int):
    system = (
        "You are a professional translation engine. "
        "Preserve meaning, tone and formatting. "
        "Do not summarize or omit content. "
        "Keep line breaks and inline markup. "
        "Output must be wrapped strictly inside <OUTPUT>...</OUTPUT> with nothing else."
    )
    user_msg = (
        f"Translate from {src_lang} to {tgt_lang}. "
        "Keep all formatting exactly.\n"
        "<TEXT>\n" + text + "\n</TEXT>"
    )
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_output_tokens,
        "messages": [{"role": "user", "content": user_msg}],
        "system": system,
    }

    tokens_in = _estimate_tokens(user_msg) + _estimate_tokens(system)
    await _anthropic_rate_gate(tokens_in=tokens_in, tokens_out_budget=max_output_tokens)

    async with httpx.AsyncClient(timeout=120) as ac:
        resp = await ac.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        out = "".join(p.get("text", "") for p in data.get("content", []) if p.get("type") == "text")

        translation = clean_translation(out)
        return translation

def _split_for_translation(long_text: str, chunk_words: int = 800, overlap_words: int = 50):
    return split_paragraph_semantic(long_text, max_words=chunk_words, overlap=overlap_words)

async def translate_with_claude(text: str, target_lang: str, source_lang: str | None = None):
    if not text.strip():
        return text
    src = source_lang or detect_lang(text) or "auto"
    tgt = target_lang or "en"
    if src == tgt or (tgt == "en" and src == "en"):
        return text

    chunks = _split_for_translation(text, chunk_words=800, overlap_words=60)
    results = []
    MAX_OUT = 3500
    for ch in chunks:
        try:
            translated = await _claude_translate_chunk(ch, src_lang=src, tgt_lang=tgt, max_output_tokens=MAX_OUT)
        except Exception:
            try:
                translated = GoogleTranslator(source="auto", target=tgt).translate(ch)
            except Exception:
                translated = ch
        results.append(translated)
    return "\n".join(results)

async def translate_to_en(text: str):
    lang = detect_lang(text)
    if USE_CLAUDE_TRANSLATION:
        translated = await translate_with_claude(text, target_lang="en", source_lang=lang)
        return translated, (lang or "en")
    # fallback đường cũ
    try:
        if lang == "en":
            return text, "en"
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated, lang
    except Exception:
        return text, (lang or "en")

async def translate_from_en(text: str, target_lang: str):
    if not target_lang or target_lang == "en":
        return text
    if USE_CLAUDE_TRANSLATION:
        return await translate_with_claude(text, target_lang=target_lang, source_lang="en")
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        return text

# ========= LightRAG init =========
async def initialize_chatbot_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=groq_model_complete,
        llm_model_name="llama-3.3-70b-versatile",
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=jina_embed
        )
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

# ========= Lưu ngôn ngữ gốc theo doc =========
DOC_LANG_MAP_PATH = os.path.join(WORKING_DIR, "doc_lang_map.json")
if not os.path.exists(DOC_LANG_MAP_PATH):
    with open(DOC_LANG_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump({}, f)

def save_doc_lang(doc_id: str, lang: str):
    try:
        with open(DOC_LANG_MAP_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        m = {}
    m[doc_id] = lang
    with open(DOC_LANG_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

def load_doc_lang_map():
    try:
        with open(DOC_LANG_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# ========= Ingestion: dịch sang EN bằng Claude =========
async def add_document_to_rag(file, rag: LightRAG):
    try:
        raw_text = read_file_text(file)
        raw_text = normalize_text(raw_text)
        if not raw_text:
            return "⚠️ File rỗng hoặc không đọc được nội dung.", False

        import hashlib, time as _t
        doc_id = hashlib.sha1((file.name + str(_t.time())).encode()).hexdigest()[:12]

        sample = raw_text[:2000]
        doc_lang = detect_lang(sample) or "en"
        print(f"[ingest] detected doc_lang={doc_lang}")

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
        success = 0
        for p in paragraphs:
            chunks = split_paragraph_semantic(p, max_words=300, overlap=50)
            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                # Dịch sang EN bằng Claude
                chunk_en, _orig = await translate_to_en(chunk)
                try:
                    await rag.ainsert(chunk_en, metadata={"doc_id": doc_id, "orig_lang": doc_lang})
                except TypeError:
                    await rag.ainsert(chunk_en)
                success += 1

        save_doc_lang(doc_id, doc_lang)
        return f"✅ Đã thêm {success} chunk. doc_id={doc_id}, lang={doc_lang}", True
    except Exception as e:
        return f"❌ Lỗi khi xử lý file: {e}", False

# ========= Query / Chat =========
RESPONSE_RULES = (
    "You must strictly extract information from the inserted documents. "
    "You may paraphrase and summarize only if the meaning is preserved. "
    "If the documents do not contain relevant information, reply exactly: 'No information found in the document.' "
    "Do not include a References section or explicitly list sources. Do not mention internal system names, metadata, or the knowledge graph."
    "If multiple retrieved passages have similar meaning, keep only the earliest one."
)

def to_light_history(hist):
    out = []
    for item in (hist or []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user, assistant = item
            if user:
                out.append({"role": "user", "content": user})
            if assistant:
                out.append({"role": "assistant", "content": assistant})
    return out


def clean_for_chatbot(text: str) -> str:
    if not text:
        return ""
    # Thêm \n trước mọi heading (#, ##, ###...)
    text = re.sub(r'(\s*)(#{1,6}\s*)', r'\n\2', text)
    # Thêm \n trước gạch đầu dòng
    text = re.sub(r'(\s*)([-*]\s+)', r'\n\2', text)
    # Chuẩn hóa nhiều dòng trống
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

# ========= Query / Chat =========
async def RAG_chatbot(message, history, rag, file, query_mode, translate_back=True):
    history = history or []
    user_lang = detect_lang(message) or "en"

    # 1) Dịch câu hỏi sang EN
    message_en, _src = await translate_to_en(message)

    # 2) Embed detected language
    query_en = f"[DETECTED_LANGUAGE={user_lang}] {message_en}"
    prompt = f"{RESPONSE_RULES}\n\nTopic: {query_en}"

    qp = QueryParam(
        mode=query_mode,
        top_k=10,
        user_prompt=prompt,
        conversation_history=to_light_history(history),
    )

    # 3) Query RAG
    response_en = await rag.aquery(query_en, qp)

    # 4) Dịch ngược
    if translate_back and user_lang != "en":
        try:
            final = await translate_from_en(response_en, user_lang)
        except Exception:
            final = response_en
    else:
        final = response_en

    history = history or []

    history.append((message, clean_for_chatbot(final)))

    return history, history

def create_rag_wrapper(rag):
    async def rag_wrapper(message, history, file, query_mode):
        return await RAG_chatbot(message, history, rag, file, query_mode)
    return rag_wrapper
# ========= Gradio UI =========
async def main():
    rag = await initialize_chatbot_rag()

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("## 🌍 Multilingual LightRAG")
            with gr.Column(scale=1):
                query_mode = gr.Dropdown(
                    choices=["local", "hybrid"],
                    value="hybrid",
                    label="⚙️ Mode",
                )

        rag_state = gr.State(rag)

        chatbot_ui = gr.Chatbot(type="messages", label="💬 Hội thoại")
        msg = gr.Textbox(label="💬 Nhập câu hỏi")
        history = gr.State([])

        with gr.Row():
            send_btn = gr.Button("📩 Gửi")
            upload_btn = gr.Button("📥 Thêm tài liệu")

        file_input = gr.File(label="📄 Upload file .txt/.docx", file_types=[".txt", ".docx"])
        upload_result = gr.Textbox(label="Kết quả thêm tài liệu", interactive=False)

        upload_btn.click(
            fn=add_document_to_rag,
            inputs=[file_input, rag_state],
            outputs=[upload_result, gr.State(True)],
        )

        rag_wrapper_func = create_rag_wrapper(rag)

        send_btn.click(
            fn=rag_wrapper_func,
            inputs=[msg, history, file_input, query_mode],
            outputs=[chatbot_ui, history],
        )

        msg.submit(
            fn=rag_wrapper_func,
            inputs=[msg, history, file_input, query_mode],
            outputs=[chatbot_ui, history],
        )

        demo.launch(server_name="127.0.0.1", server_port=9621, show_api=False, share=True)

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())