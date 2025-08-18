import os
import logging
import json
import httpx
import base64
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Depends, Cookie, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()
import asyncio
import io
import database

# Логирование
logging.basicConfig(level=logging.INFO)

# Инициализация FastAPI приложения и CORS
app = FastAPI(title="explAiner API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение статических файлов (если нужны картинки, css, js)
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# Подключение шаблонов (HTML)
templates = Jinja2Templates(directory="templates")

# 🔹 Главная страница (рендерит HTML)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("new.html", {"request": request})

# 🔹 Health-check (Render будет проверять этот URL)
@app.get("/status")
async def status():
    return {"status": "ok", "service": "explAiner AI API"}

# 🔹 Альтернативные роуты для прямого доступа к HTML
@app.get("/app")
async def serve_app():
    return FileResponse("templates/new.html")

@app.get("/ui")
async def serve_ui():
    return FileResponse("templates/new.html")


class ChatRequest(BaseModel):
    message: str
    model: str = "gpt-4o"
    multilingual: bool = True
    factCheck: bool = True
    rag: Optional[Dict[str, Any]] = None
    user_id: Optional[int] = None
    chat_id: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # "user" или "assistant"
    content: str
    timestamp: str
    mode: str = "general"


class UserRegister(BaseModel):
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class ChatHistory(BaseModel):
    id: str
    title: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str


# Интеграция с Groq API (с безопасным фолбэком)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


def load_chat_history() -> List[Dict[str, Any]]:
    """Загрузка истории чата из файла"""
    try:
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"Ошибка загрузки истории чата: {e}")
    return []


def save_chat_history(history: List[Dict[str, Any]]):
    """Сохранение истории чата в файл"""
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Ошибка сохранения истории чата: {e}")


def generate_chat_id() -> str:
    """Генерация уникального ID для чата"""
    return f"chat_{int(datetime.now().timestamp())}"


def generate_chat_title(query: str) -> str:
    """Генерация заголовка чата на основе первого сообщения"""
    # Удаляем переносы строк и лишние пробелы
    query = query.replace('\n', ' ').strip()
    
    # Если запрос короткий, используем его целиком
    if len(query) <= 50:
        return query
    
    # Пытаемся найти первое предложение или вопрос
    sentence_end = None
    for end_char in ['.', '?', '!']:
        pos = query.find(end_char)
        if pos > 10 and pos <= 100:  # Предложение должно быть не слишком коротким и не слишком длинным
            sentence_end = pos + 1
            break
    
    if sentence_end:
        # Используем первое предложение
        return query[:sentence_end].strip()
    
    # Если не нашли подходящего предложения, ищем конец фразы
    for delimiter in [',', ';', ':', ' - ', ' и ', ' или ']:
        pos = query.find(delimiter)
        if pos > 15 and pos <= 70:  # Фраза должна быть не слишком короткой и не слишком длинной
            return query[:pos].strip()
    
    # Если не нашли подходящих разделителей, обрезаем по словам
    words = query.split()
    title = ""
    for word in words:
        if len(title) + len(word) + 1 > 50:
            break
        title += word + " "
    
    # Если получилось слишком коротко, используем обычное обрезание
    if len(title.strip()) < 20:
        return query[:47] + "..."
        
    return title.strip()


async def call_groq(prompt: str, model: str = "gpt-4o", multilingual: bool = True, factCheck: bool = True) -> str:
    """Вызов Groq API. При отсутствии ключа возвращает фолбэк-ответ.

    Для продакшена укажите переменную окружения GROQ_API_KEY.
    """
    if not GROQ_API_KEY:
        # Фолбэк: локальный ответ без внешней LLM
        return generate_fallback_response(prompt)

    # Системный промпт для explAiner
    system_prompt = "Ты explAiner - юридический AI-ассистент. Отвечай кратко и по делу на русском языке (или на языке пользователя, если включен multilingual). Если не знаешь ответа, честно скажи об этом. Используй markdown для форматирования."
    
    if factCheck:
        system_prompt += " Проверяй факты и указывай источники информации, когда это возможно."

    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama3-70b-8192" if model == "gpt-4o" else "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2000,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content or "Не удалось получить ответ от ИИ."
    except Exception as e:
        logging.error(f"Groq API error: {e}")
        return generate_fallback_response(prompt)


def generate_fallback_response(prompt: str, mode: str = "general") -> str:
    """Генерирует локальный ответ без внешней LLM"""
    
    # Простые правила для разных режимов
    if mode == "contract":
        if any(word in prompt.lower() for word in ["договор", "контракт", "соглашение"]):
            return """📋 Анализ договора (локальный режим)

🔍 Основные моменты для проверки:
• Предмет договора и обязательства сторон
• Сроки исполнения и порядок расчетов
• Ответственность и штрафные санкции
• Порядок расторжения и изменения
• Конфиденциальность и интеллектуальная собственность

⚠️ Рекомендации:
1. Проверьте баланс ответственности сторон
2. Обратите внимание на односторонние изменения условий
3. Убедитесь в четкости формулировок
4. Проверьте соответствие законодательству

💡 Для детального анализа подключите GROQ_API_KEY"""
        
    elif mode == "legal":
        return """⚖️ Юридическая консультация (локальный режим)

📚 Общие принципы:
• Всегда проверяйте актуальность законодательства
• Учитывайте судебную практику
• Документируйте все соглашения письменно
• Консультируйтесь с профильными юристами

🔗 Полезные ресурсы:
• КонсультантПлюс
• Гарант
• Официальные сайты ведомств

💡 Для получения конкретных рекомендаций подключите GROQ_API_KEY"""
        
    elif mode == "summary":
        return """📊 Краткая сводка (локальный режим)

📋 Что анализировать:
• Ключевые условия и обязательства
• Потенциальные риски и спорные моменты
• Финансовые последствия
• Сроки и порядок исполнения

🎯 Рекомендации:
• Выделите критически важные пункты
• Оцените риски по шкале 1-10
• Подготовьте вопросы для согласования
• Документируйте все замечания

💡 Для детального анализа подключите GROQ_API_KEY"""
    
    else:
        return f"""🤖 Ответ ИИ (локальный режим)

Ваш запрос: {prompt[:100]}{'...' if len(prompt) > 100 else ''}

💡 Для получения полноценного ответа от ИИ подключите GROQ_API_KEY в переменных окружения.

🔧 Как подключить:
1. Получите API ключ на https://console.groq.com/
2. Установите переменную окружения: GROQ_API_KEY=ваш_ключ
3. Перезапустите приложение

📚 Пока что я работаю в демо-режиме с базовыми ответами."""


@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    """Чат с ИИ с сохранением истории"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Запрос не может быть пустым")
        
        # Получаем ответ от ИИ
        answer = await call_groq(request.message, request.model, request.multilingual, request.factCheck)
        
        # Если указан user_id, сохраняем в базе данных
        if request.user_id:
            # Используем существующий chat_id или создаем новый
            chat_id = request.chat_id or generate_chat_id()
            
            # Получаем существующий чат или создаем новый
            chat_data = database.get_chat(request.user_id, chat_id)
            
            if chat_data["success"]:
                # Чат существует
                messages = chat_data["chat"]["messages"]
                title = chat_data["chat"]["title"]
            else:
                # Создаем новый чат
                messages = []
                title = generate_chat_title(request.message)
            
            # Добавляем сообщения
            messages.append({
                "role": "user",
                "content": request.message,
                "timestamp": datetime.now().isoformat()
            })
            
            messages.append({
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().isoformat()
            })
            
            # Сохраняем в базу
            database.save_chat(request.user_id, chat_id, title, messages)
            
            # Возвращаем ответ с chat_id
            return JSONResponse(content={
                "answer": answer,
                "chat_id": chat_id,
                "title": title
            })
        else:
            # Обратная совместимость - сохраняем в файл
            history = load_chat_history()
            
            # Создаем новый чат
            chat_id = generate_chat_id()
            new_chat = {
                "id": chat_id,
                "title": generate_chat_title(request.message),
                "messages": [],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            history.append(new_chat)
            current_chat = new_chat
            
            # Добавляем сообщение пользователя
            user_message = {
                "role": "user",
                "content": request.message,
                "timestamp": datetime.now().isoformat()
            }
            current_chat["messages"].append(user_message)
            
            # Добавляем ответ ассистента
            assistant_message = {
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().isoformat()
            }
            current_chat["messages"].append(assistant_message)
            
            # Обновляем время последнего изменения
            current_chat["updated_at"] = datetime.now().isoformat()
            
            # Сохраняем историю
            save_chat_history(history)
            
            # Возвращаем ответ
            return answer
        
    except Exception as e:
        logging.error(f"Ошибка генерации ответа: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации ответа: {str(e)}")


@app.get("/chats")
async def get_chat_history(user_id: Optional[int] = None):
    """Получение списка всех чатов"""
    try:
        if user_id:
            # Получаем чаты из базы данных
            result = database.get_user_chats(user_id)
            if result["success"]:
                return JSONResponse(content={"chats": result["chats"]})
            else:
                raise HTTPException(status_code=500, detail=result["message"])
        else:
            # Обратная совместимость - получаем из файла
            history = load_chat_history()
            return JSONResponse(content={"chats": history})
    except Exception as e:
        logging.error(f"Ошибка получения истории чата: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения истории: {str(e)}")


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str, user_id: Optional[int] = None):
    """Получение конкретного чата по ID"""
    try:
        if user_id:
            # Получаем чат из базы данных
            result = database.get_chat(user_id, chat_id)
            if result["success"]:
                return JSONResponse(content=result["chat"])
            else:
                raise HTTPException(status_code=404, detail="Чат не найден")
        else:
            # Обратная совместимость - получаем из файла
            history = load_chat_history()
            for chat in history:
                if chat["id"] == chat_id:
                    return JSONResponse(content=chat)
            raise HTTPException(status_code=404, detail="Чат не найден")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Ошибка получения чата: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения чата: {str(e)}")


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, user_id: Optional[int] = None):
    """Удаление чата"""
    try:
        if user_id:
            # Удаляем чат из базы данных
            result = database.delete_chat(user_id, chat_id)
            if result["success"]:
                return JSONResponse(content={"message": "Чат удален", "chat_id": chat_id})
            else:
                raise HTTPException(status_code=404, detail=result["message"])
        else:
            # Обратная совместимость - удаляем из файла
            history = load_chat_history()
            history = [chat for chat in history if chat["id"] != chat_id]
            save_chat_history(history)
            return JSONResponse(content={"message": "Чат удален", "chat_id": chat_id})
    except Exception as e:
        logging.error(f"Ошибка удаления чата: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка удаления чата: {str(e)}")


@app.post("/chats/{chat_id}/title")
async def update_chat_title(chat_id: str, title: str, user_id: Optional[int] = None):
    """Обновление заголовка чата"""
    try:
        if user_id:
            # Обновляем заголовок в базе данных
            result = database.update_chat_title(user_id, chat_id, title)
            if result["success"]:
                return JSONResponse(content={"message": "Заголовок обновлен", "title": title})
            else:
                raise HTTPException(status_code=404, detail=result["message"])
        else:
            # Обратная совместимость - обновляем в файле
            history = load_chat_history()
            for chat in history:
                if chat["id"] == chat_id:
                    chat["title"] = title
                    chat["updated_at"] = datetime.now().isoformat()
                    save_chat_history(history)
                    return JSONResponse(content={"message": "Заголовок обновлен", "title": title})
            raise HTTPException(status_code=404, detail="Чат не найден")
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Ошибка обновления заголовка: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обновления заголовка: {str(e)}")


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return JSONResponse(content={
        "status": "healthy",
        "groq_available": bool(GROQ_API_KEY),
        "service": "ExplAiner AI"
    })


@app.get("/stats")
async def get_stats():
    """Статистика сервиса"""
    try:
        history = load_chat_history()
        total_chats = len(history)
        total_messages = sum(len(chat["messages"]) for chat in history)
        
        return JSONResponse(content={
            "service": "ExplAiner AI",
            "version": "1.0.0",
            "groq_configured": bool(GROQ_API_KEY),
            "modes_available": ["general", "contract", "legal", "summary"],
            "total_chats": total_chats,
            "total_messages": total_messages
        })
    except Exception as e:
        logging.error(f"Ошибка получения статистики: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики: {str(e)}")


# Инициализация при запуске приложения
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    logging.info("ExplAiner AI система инициализирована")
    if not GROQ_API_KEY:
        logging.warning("GROQ_API_KEY не установлен. Работаем в локальном режиме.")


# Обработка загрузки файлов
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), encrypted: str = Form("0")):
    """Загрузка файла"""
    try:
        # Создаем директорию для загруженных файлов, если её нет
        os.makedirs("uploads", exist_ok=True)
        
        # Сохраняем файл
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "size": len(content),
            "encrypted": encrypted == "1"
        })
    except Exception as e:
        logging.error(f"Ошибка загрузки файла: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки файла: {str(e)}")


class AudioRequest(BaseModel):
    text: str
    voice: Optional[str] = None


# Генерация аудио (заглушка)
@app.post("/api/audio")
async def generate_audio(request: AudioRequest):
    """Генерация аудио из текста с использованием gTTS"""
    try:
        from gtts import gTTS
        import tempfile
        import os
        
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_path = temp_file.name
        
        # Генерируем аудио с помощью gTTS
        tts = gTTS(text=request.text, lang='ru' if not request.voice else 'en')
        tts.save(temp_path)
        
        # Возвращаем аудио файл
        def cleanup():
            try:
                os.unlink(temp_path)
            except:
                pass
                
        return FileResponse(
            temp_path, 
            media_type="audio/mpeg",
            background=cleanup
        )
    except ImportError:
        logging.warning("gTTS не установлен. Возвращаем пустой MP3.")
        return FileResponse("templates/empty.mp3", media_type="audio/mpeg")
    except Exception as e:
        logging.error(f"Ошибка генерации аудио: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации аудио: {str(e)}")


class VideoRequest(BaseModel):
    text: str
    avatar: Optional[str] = "none"
    voice: Optional[str] = None


class ComplianceRequest(BaseModel):
    text: str
    profiles: List[str] = ["GDPR"]


class ComplianceResponse(BaseModel):
    score: int
    issues: List[str]
    suggestions: List[str]


class CompareRequest(BaseModel):
    doc_a: str
    doc_b: str


class CompareResponse(BaseModel):
    diffs: List[Dict[str, Any]]
    summary: str


class WhatIfRequest(BaseModel):
    question: str
    context: Optional[str] = None


class WhatIfResponse(BaseModel):
    diagram: str
    explanation: str


class DocumentAnalysisRequest(BaseModel):
    document_id: str
    document_text: str
    analysis_type: str = "general"  # general, legal, risks


class DocumentAnalysisResponse(BaseModel):
    summary: str
    key_points: List[str]
    risks: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None


# Генерация видео (заглушка)
@app.post("/api/video")
async def generate_video(request: VideoRequest):
    try:
        HEYGEN_API_KEY = os.getenv('HEYGEN_API_KEY')
        if not HEYGEN_API_KEY:
            raise ValueError("HeyGen API key not found")

        logging.info(f"Generating video with text: {request.text[:50]}...")

        url = "https://api.heygen.com/v1/video.generate"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": HEYGEN_API_KEY
        }
        payload = {
            "test": True,
            "caption": False,
            "aspect_ratio": "16:9",
            "video_inputs": [
                {
                    "character": {
                        "type": "avatar",
                        "avatar_name": request.avatar or "Jessica-2k-20190523"  # Replace with your valid avatar name
                    },
                    "voice": {
                        "type": "text",
                        "input_text": request.text,
                        "voice_name": request.voice or "Sara-neutral"  # Replace with valid voice name
                    }
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=payload)
            logging.info(f"Generation response status: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            logging.info(f"Generation data: {data}")
            video_id = data.get("data", {}).get("video_id")
            if not video_id:
                raise ValueError("No video_id in response")

            # Poll for status
            status_url = f"https://api.heygen.com/v1/video_status/{video_id}"
            attempts = 0
            max_attempts = 60  # 5 min
            while attempts < max_attempts:
                attempts += 1
                status_resp = await client.get(status_url, headers=headers)
                logging.info(f"Status check {attempts}: {status_resp.status_code}")
                status_resp.raise_for_status()
                status_data = status_resp.json()
                logging.info(f"Status data: {status_data}")
                status = status_data.get("data", {}).get("status")
                if status == "completed":
                    video_url = status_data.get("data", {}).get("video_url")
                    break
                elif status == "failed" or status == "error":
                    raise ValueError(f"Video generation failed: {status_data.get('data', {}).get('error_msg', 'Unknown error')}")
                await asyncio.sleep(5)

            if not video_url:
                raise ValueError("Video generation timed out")

            # Download video
            video_resp = await client.get(video_url)
            logging.info(f"Download response: {video_resp.status_code}")
            video_resp.raise_for_status()

            return StreamingResponse(io.BytesIO(video_resp.content), media_type="video/mp4")

    except Exception as e:
        logging.error(f"Video generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video generation error: {str(e)}")


# Комплаенс-чекер
@app.post("/api/compliance")
async def check_compliance(request: ComplianceRequest):
    """Проверка текста на соответствие требованиям GDPR, CCPA и других стандартов"""
    try:
        if not request.text or request.text.strip() == "":
            return JSONResponse(content={
                "score": 0,
                "issues": ["Текст не предоставлен"],
                "suggestions": ["Пожалуйста, предоставьте текст для проверки"]
            })
            
        text = request.text.lower()
        profiles = request.profiles
        issues = []
        suggestions = []
        
        # Базовые проверки для GDPR
        if "gdpr" in profiles:
            if not any(term in text for term in ["consent", "согласие", "соглас"]):
                issues.append("Нет явного основания обработки (consent/contract).")
                suggestions.append("Добавьте правовое основание обработки (Art.6 GDPR) и цели.")
                
            if any(term in text for term in ["third party", "third-party", "третьим лицам", "третьих лиц"]):
                issues.append("Указана передача третьим лицам — требуется DPA и SCC при трансграничной передаче.")
                suggestions.append("Заключите DPA с процессорами и укажите меры безопасности (Art.28).")
                
            if not any(term in text for term in ["retention", "срок хранения", "период хранения"]):
                issues.append("Не указан срок хранения.")
                suggestions.append("Укажите сроки хранения и критерии их определения.")
                
            if not any(term in text for term in ["rights", "права субъекта", "право на доступ"]):
                issues.append("Нет описания прав субъекта (доступ, удаление, перенос).")
                suggestions.append("Опишите права субъекта и порядок их реализации.")
        
        # Проверки для CCPA
        if "ccpa" in profiles:
            if not any(term in text for term in ["california", "калифорния", "ccpa"]):
                issues.append("Нет упоминания CCPA для пользователей из Калифорнии.")
                suggestions.append("Добавьте раздел о правах резидентов Калифорнии согласно CCPA.")
                
            if not any(term in text for term in ["opt-out", "отказаться", "отказ от"]):
                issues.append("Нет механизма отказа от продажи данных (Do Not Sell).")
                suggestions.append("Добавьте механизм Do Not Sell My Personal Information.")
        
        # Проверки для 152-ФЗ (Россия)
        if "152fz" in profiles:
            if not any(term in text for term in ["оператор", "обработка персональных данных", "субъект персональных данных"]):
                issues.append("Отсутствуют основные термины согласно 152-ФЗ.")
                suggestions.append("Добавьте определения оператора, субъекта ПДн и процессов обработки.")
                
            if not any(term in text for term in ["согласие на обработку", "отзыв согласия"]):
                issues.append("Не описан порядок получения и отзыва согласия.")
                suggestions.append("Укажите порядок получения и отзыва согласия на обработку ПДн.")
        
        # Расчет оценки соответствия
        score = max(0, 100 - len(issues)*15)  # Снижаем штраф за каждую проблему
        
        return JSONResponse(content={
            "score": score,
            "issues": issues,
            "suggestions": list(set(suggestions))  # Убираем дубликаты
        })
    except Exception as e:
        logging.error(f"Ошибка проверки комплаенса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка проверки комплаенса: {str(e)}")


# Сравнение документов
@app.post("/api/compare")
async def compare_documents(request: CompareRequest):
    """Сравнение двух документов и выявление различий"""
    try:
        # В реальном приложении здесь была бы более сложная логика сравнения
        # С использованием библиотек для работы с документами разных форматов
        
        # Простое сравнение текста
        import difflib
        
        doc_a = request.doc_a
        doc_b = request.doc_b
        
        # Используем difflib для сравнения
        diff = difflib.ndiff(doc_a.splitlines(), doc_b.splitlines())
        
        # Формируем результат
        diffs = []
        for line in diff:
            if line.startswith('+ '):
                diffs.append({'type': 'added', 'text': line[2:]})
            elif line.startswith('- '):
                diffs.append({'type': 'removed', 'text': line[2:]})
            elif line.startswith('? '):
                # Игнорируем метаданные
                continue
            else:
                diffs.append({'type': 'unchanged', 'text': line[2:] if line.startswith('  ') else line})
        
        # Анализ различий
        added = len([d for d in diffs if d['type'] == 'added'])
        removed = len([d for d in diffs if d['type'] == 'removed'])
        total_changes = added + removed
        
        # Формируем краткий отчет
        summary = f"Найдено {total_changes} изменений: {added} добавлений, {removed} удалений."
        
        return JSONResponse(content={
            "diffs": diffs[:500],  # Ограничиваем размер ответа
            "summary": summary
        })
    except Exception as e:
        logging.error(f"Ошибка сравнения документов: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка сравнения документов: {str(e)}")


# What-if симулятор
@app.post("/api/whatif")
async def generate_whatif(request: WhatIfRequest):
    """Генерация Mermaid-диаграмм для What-if сценариев"""
    try:
        if not request.question or request.question.strip() == "":
            return JSONResponse(status_code=400, content={
                "error": "Вопрос не может быть пустым",
                "diagram": "",
                "explanation": "Пожалуйста, укажите вопрос для анализа"
            })
        
        # Проверяем наличие контекста документа
        if not request.context or request.context.strip() == "":
            return JSONResponse(content={
                "diagram": """flowchart TD
                A["What-If Симулятор"] --> B{{"Требуется документ"}}
                B --> C["Загрузите документ"]
                C --> D["Для использования What-If симулятора"]
                D --> E["сначала загрузите документ для анализа"]
                """,
                "explanation": "Для использования What-If симулятора необходимо сначала загрузить документ. Пожалуйста, загрузите документ через панель загрузки файлов."
            })
        
        # Проверка на наличие API ключа
        if not GROQ_API_KEY:
            # Заглушка для what-if симулятора
            return JSONResponse(content={
                "diagram": """flowchart TD
                A["What-If Симулятор (демо-режим)"] --> B{{"Для полной версии требуется API ключ"}}
                B --> C["Базовая диаграмма"]
                C --> D["Для активации полной версии"]
                D --> E["Укажите GROQ_API_KEY в переменных окружения"]
                """,
                "explanation": "Это демонстрационная версия What-If симулятора. Для доступа к полной функциональности необходимо указать GROQ_API_KEY в переменных окружения."
            })
            
        question = request.question
        context = request.context or ""
        
        # Выбираем шаблон в зависимости от вопроса
        diagram = ""
        explanation = ""
        
        # Юридические сценарии
        if any(term in question.lower() for term in ["gdpr", "персонал", "personal data", "данные"]):
            diagram = """flowchart TD
            A[Обработка персональных данных] --> B{Есть основание?}
            B -->|Consent/Contract| C[Определить цели и объем]
            B -->|Нет| H[Запрет/Ограничить]
            C --> D{Передача за пределы ЕЭЗ?}
            D -->|Да| E[SCC/Адекватность]
            D -->|Нет| F[Локальная обработка]
            C --> G[Права субъекта, сроки хранения]"""
            explanation = "Диаграмма показывает процесс обработки персональных данных в соответствии с GDPR. Ключевые требования: наличие правового основания, особые условия для трансграничной передачи, обеспечение прав субъектов."
            
        elif any(term in question.lower() for term in ["наруш", "breach", "неконкурен", "non-compete", "договор", "контракт"]):
            diagram = """flowchart TD
            A[Нарушение договора] --> B{Уведомление контрагента?}
            B -->|Да| C[Попытка урегулирования]
            B -->|Нет| D[Расторжение/штраф]
            C --> E{Согласие достигнуто?}
            E -->|Да| F[Доп. соглашение]
            E -->|Нет| D
            D --> G[Судебная перспектива]"""
            explanation = "Диаграмма показывает возможные сценарии при нарушении договора. В случае нарушения пункта о неконкуренции возможны штрафные санкции, судебные разбирательства и запрет на ведение деятельности."
            
        # Корпоративные сценарии
        elif any(term in question.lower() for term in ["компания", "корпоратив", "акционер", "устав", "company", "corporate"]):
            diagram = """flowchart TD
            A[Корпоративное решение] --> B{Требуется согласие?}
            B -->|Совет директоров| C[Созыв заседания]
            B -->|Общее собрание| D[Созыв ОСА]
            B -->|Нет| E[Решение исп. органа]
            C --> F[Протокол СД]
            D --> G[Протокол ОСА]
            F --> H[Исполнение решения]
            G --> H
            E --> H"""
            explanation = "Диаграмма показывает процесс принятия корпоративных решений в зависимости от компетенции органов управления. Для разных типов решений требуется одобрение соответствующего органа."
            
        # Налоговые сценарии
        elif any(term in question.lower() for term in ["налог", "tax", "ндс", "налогообложение"]):
            diagram = """flowchart TD
            A[Налоговый сценарий] --> B{Тип операции?}
            B -->|Внутренняя| C[Стандартный учет]
            B -->|Международная| D[Проверка на BEPS]
            C --> E[Налоговые риски]
            D --> F{Есть соглашение?}
            F -->|Да| G[Применение СОИДН]
            F -->|Нет| H[Общий порядок]
            G --> I[Отчетность]
            H --> I
            E --> I"""
            explanation = "Диаграмма показывает процесс налогового планирования и учета в зависимости от типа операций. Для международных операций важно учитывать наличие соглашений об избежании двойного налогообложения."
            
        # Общий шаблон для других сценариев
        else:
            # Создаем более детальную диаграмму на основе ключевых слов в вопросе
            keywords = ["риск", "срок", "ответственность", "санкция", "штраф", "суд", "арбитраж", "спор"]
            scenario = question[:40] + '...' if len(question) > 40 else question
            
            # Определяем тип сценария
            scenario_type = "Юридический сценарий"
            for word in ["бизнес", "компания", "продажи", "маркетинг"]:
                if word in question.lower():
                    scenario_type = "Бизнес-сценарий"
                    break
                    
            diagram = f"""flowchart TD
            A["{scenario_type}: {scenario}"] --> B{{Оценка рисков}}
            B -->|Высокий| C[Детальный анализ]
            B -->|Средний| D[Мониторинг]
            B -->|Низкий| E[Стандартная процедура]
            C --> F[Юридическая консультация]
            C --> G[План снижения рисков]
            D --> H[Контрольные точки]
            F --> I[Принятие решения]
            G --> I
            H --> I
            E --> I"""
            
            explanation = f"Диаграмма показывает процесс анализа и принятия решений для сценария: '{question}'. После оценки рисков применяются различные стратегии в зависимости от их уровня, с последующим контролем и принятием окончательного решения."
        
        return JSONResponse(content={
            "diagram": diagram,
            "explanation": explanation
        })
    except Exception as e:
        logging.error(f"Ошибка генерации What-if диаграммы: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации What-if диаграммы: {str(e)}")


# Анализ документов
@app.post("/api/analyze")
async def analyze_document(request: DocumentAnalysisRequest):
    """Анализ документа с выделением ключевых моментов, рисков и рекомендаций"""
    try:
        document_text = request.document_text
        analysis_type = request.analysis_type
        
        # В реальном приложении здесь был бы вызов LLM для анализа документа
        # В демо используем шаблонный анализ
        
        # Базовый анализ документа
        summary = "Документ содержит примерно {len(document_text.split())} слов и относится к юридической документации."
        key_points = [
            "Определены основные стороны договора и их обязательства",
            "Указаны сроки исполнения и условия оплаты",
            "Присутствуют положения о конфиденциальности"
        ]
        risks = []
        recommendations = []
        
        # Дополнительный анализ в зависимости от типа
        if analysis_type == "legal":
            key_points.extend([
                "Прописаны правовые основания и юрисдикция",
                "Указан порядок разрешения споров"
            ])
        elif analysis_type == "risks":
            risks = [
                "Нечетко прописаны условия ответственности сторон",
                "Отсутствуют четкие критерии приемки работ",
                "Не прописан порядок изменения условий договора"
            ]
            recommendations = [
                "Уточнить пределы ответственности сторон",
                "Добавить четкие критерии приемки работ",
                "Прописать порядок изменения условий договора"
            ]
        
        # Анализ ключевых слов в документе
        text_lower = document_text.lower()
        if "gdpr" in text_lower or "персональн" in text_lower:
            key_points.append("Содержит положения о персональных данных")
            if analysis_type == "risks":
                risks.append("Требуется проверка на соответствие GDPR")
                recommendations.append("Провести комплаенс-чек по GDPR")
        
        if "неконкурен" in text_lower or "non-compete" in text_lower:
            key_points.append("Содержит положения о неконкуренции")
            if analysis_type == "risks":
                risks.append("Требуется проверка положений о неконкуренции на соответствие законодательству")
        
        return JSONResponse(content={
            "summary": summary,
            "key_points": key_points,
            "risks": risks,
            "recommendations": recommendations
        })
    except Exception as e:
        logging.error(f"Ошибка анализа документа: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка анализа документа: {str(e)}")


# Регистрация пользователя
@app.post("/api/register")
async def register(user: UserRegister):
    """Регистрация нового пользователя"""
    try:
        result = database.register_user(user.username, user.email, user.password)
        
        if result["success"]:
            return JSONResponse(content={
                "success": True,
                "user": {
                    "id": result["user_id"],
                    "username": result["username"],
                    "email": result["email"],
                    "created_at": result["created_at"]
                }
            })
        else:
            return JSONResponse(status_code=400, content={
                "success": False,
                "message": result["message"]
            })
    except Exception as e:
        logging.error(f"Ошибка регистрации: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка регистрации: {str(e)}")


# Авторизация пользователя
@app.post("/api/login")
async def login(user: UserLogin):
    """Авторизация пользователя"""
    try:
        result = database.login_user(user.email, user.password)
        
        if result["success"]:
            return JSONResponse(content={
                "success": True,
                "user": {
                    "id": result["user_id"],
                    "username": result["username"],
                    "email": result["email"],
                    "created_at": result["created_at"]
                }
            })
        else:
            return JSONResponse(status_code=401, content={
                "success": False,
                "message": result["message"]
            })
    except Exception as e:
        logging.error(f"Ошибка авторизации: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка авторизации: {str(e)}")


# Проверка текущего пользователя
@app.get("/api/user")
async def get_current_user(user_id: int):
    """Получение информации о текущем пользователе"""
    try:
        conn = sqlite3.connect(database.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, username, email, created_at FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return JSONResponse(status_code=404, content={"success": False, "message": "Пользователь не найден"})
        
        return JSONResponse(content={
            "success": True,
            "user": {
                "id": user[0],
                "username": user[1],
                "email": user[2],
                "created_at": user[3]
            }
        })
    except Exception as e:
        logging.error(f"Ошибка получения данных пользователя: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения данных пользователя: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)