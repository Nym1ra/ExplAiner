import os
from sentence_transformers import SentenceTransformer
import numpy as np
import sqlite3
import json

def build_vector_db():
    """Создание векторной базы данных для документов"""
    print("Начинаю создание векторной базы данных...")
    
    # Инициализация модели для эмбеддингов
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("[✓] Модель загружена")
    except Exception as e:
        print(f"[!] Ошибка загрузки модели: {e}")
        return
    
    # Создание базы данных
    conn = sqlite3.connect('vectors.db')
    cursor = conn.cursor()
    
    # Создание таблицы для векторов
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            vector BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    path = "kodeks"
    processed_count = 0
    
    for filename in os.listdir(path):
        if not filename.endswith(".txt"):
            continue
            
        full_path = os.path.join(path, filename)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Разбиваем на чанки (упрощенная версия)
            chunks = [content[i:i+1000] for i in range(0, len(content), 950)]
            
            for chunk in chunks:
                if len(chunk.strip()) < 50:  # Пропускаем слишком короткие чанки
                    continue
                    
                # Создаем эмбеддинг
                vector = model.encode(chunk)
                vector_blob = vector.tobytes()
                
                # Сохраняем в базу
                cursor.execute('''
                    INSERT INTO document_vectors (filename, content, vector)
                    VALUES (?, ?, ?)
                ''', (filename, chunk, vector_blob))
                
                processed_count += 1
                
            print(f"[✓] Обработан: {filename}")
            
        except Exception as e:
            print(f"[!] Ошибка при обработке {filename}: {e}")
    
    conn.commit()
    conn.close()
    print(f"[✓] Векторная база создана! Обработано чанков: {processed_count}")

if __name__ == "__main__":
    build_vector_db()
