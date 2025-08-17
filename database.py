import sqlite3
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Настройка логирования
logging.basicConfig(level=logging.INFO)

# Путь к базе данных
DB_PATH = "users.db"

def init_db():
    """Инициализация базы данных"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Создание таблицы пользователей
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Создание таблицы для истории чатов
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        chat_id TEXT NOT NULL,
        title TEXT NOT NULL,
        messages TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    logging.info("База данных инициализирована")

def register_user(username: str, email: str, password: str) -> Dict[str, Any]:
    """Регистрация нового пользователя"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Проверка на существование пользователя с таким email
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            return {"success": False, "message": "Пользователь с таким email уже существует"}
        
        # Добавление нового пользователя
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, password)
        )
        conn.commit()
        
        # Получение ID нового пользователя
        user_id = cursor.lastrowid
        
        return {
            "success": True,
            "user_id": user_id,
            "username": username,
            "email": email,
            "created_at": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Ошибка при регистрации пользователя: {e}")
        return {"success": False, "message": f"Ошибка при регистрации: {str(e)}"}
    finally:
        conn.close()

def login_user(email: str, password: str) -> Dict[str, Any]:
    """Авторизация пользователя"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Поиск пользователя по email и паролю
        cursor.execute(
            "SELECT id, username, created_at FROM users WHERE email = ? AND password = ?",
            (email, password)
        )
        user = cursor.fetchone()
        
        if not user:
            return {"success": False, "message": "Неверный email или пароль"}
        
        return {
            "success": True,
            "user_id": user[0],
            "username": user[1],
            "email": email,
            "created_at": user[2]
        }
    except Exception as e:
        logging.error(f"Ошибка при авторизации пользователя: {e}")
        return {"success": False, "message": f"Ошибка при авторизации: {str(e)}"}
    finally:
        conn.close()

def save_chat(user_id: int, chat_id: str, title: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Сохранение истории чата"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Проверка существования чата
        cursor.execute("SELECT id FROM chat_history WHERE user_id = ? AND chat_id = ?", (user_id, chat_id))
        chat = cursor.fetchone()
        
        messages_json = json.dumps(messages, ensure_ascii=False)
        now = datetime.now().isoformat()
        
        if chat:
            # Обновление существующего чата
            cursor.execute(
                "UPDATE chat_history SET title = ?, messages = ?, updated_at = ? WHERE user_id = ? AND chat_id = ?",
                (title, messages_json, now, user_id, chat_id)
            )
        else:
            # Создание нового чата
            cursor.execute(
                "INSERT INTO chat_history (user_id, chat_id, title, messages, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, chat_id, title, messages_json, now, now)
            )
        
        conn.commit()
        return {"success": True, "chat_id": chat_id}
    except Exception as e:
        logging.error(f"Ошибка при сохранении чата: {e}")
        return {"success": False, "message": f"Ошибка при сохранении чата: {str(e)}"}
    finally:
        conn.close()

def get_user_chats(user_id: int) -> Dict[str, Any]:
    """Получение всех чатов пользователя"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT chat_id, title, messages, created_at, updated_at FROM chat_history WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        )
        chats = cursor.fetchall()
        
        result = []
        for chat in chats:
            result.append({
                "id": chat[0],
                "title": chat[1],
                "messages": json.loads(chat[2]),
                "created_at": chat[3],
                "updated_at": chat[4]
            })
        
        return {"success": True, "chats": result}
    except Exception as e:
        logging.error(f"Ошибка при получении чатов пользователя: {e}")
        return {"success": False, "message": f"Ошибка при получении чатов: {str(e)}"}
    finally:
        conn.close()

def get_chat(user_id: int, chat_id: str) -> Dict[str, Any]:
    """Получение конкретного чата пользователя"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT chat_id, title, messages, created_at, updated_at FROM chat_history WHERE user_id = ? AND chat_id = ?",
            (user_id, chat_id)
        )
        chat = cursor.fetchone()
        
        if not chat:
            return {"success": False, "message": "Чат не найден"}
        
        return {
            "success": True,
            "chat": {
                "id": chat[0],
                "title": chat[1],
                "messages": json.loads(chat[2]),
                "created_at": chat[3],
                "updated_at": chat[4]
            }
        }
    except Exception as e:
        logging.error(f"Ошибка при получении чата: {e}")
        return {"success": False, "message": f"Ошибка при получении чата: {str(e)}"}
    finally:
        conn.close()

def delete_chat(user_id: int, chat_id: str) -> Dict[str, Any]:
    """Удаление чата"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "DELETE FROM chat_history WHERE user_id = ? AND chat_id = ?",
            (user_id, chat_id)
        )
        conn.commit()
        
        if cursor.rowcount == 0:
            return {"success": False, "message": "Чат не найден"}
        
        return {"success": True, "message": "Чат успешно удален"}
    except Exception as e:
        logging.error(f"Ошибка при удалении чата: {e}")
        return {"success": False, "message": f"Ошибка при удалении чата: {str(e)}"}
    finally:
        conn.close()

def update_chat_title(user_id: int, chat_id: str, title: str) -> Dict[str, Any]:
    """Обновление заголовка чата"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "UPDATE chat_history SET title = ?, updated_at = ? WHERE user_id = ? AND chat_id = ?",
            (title, datetime.now().isoformat(), user_id, chat_id)
        )
        conn.commit()
        
        if cursor.rowcount == 0:
            return {"success": False, "message": "Чат не найден"}
        
        return {"success": True, "message": "Заголовок чата обновлен"}
    except Exception as e:
        logging.error(f"Ошибка при обновлении заголовка чата: {e}")
        return {"success": False, "message": f"Ошибка при обновлении заголовка: {str(e)}"}
    finally:
        conn.close()

# Инициализация базы данных при импорте модуля
init_db()
