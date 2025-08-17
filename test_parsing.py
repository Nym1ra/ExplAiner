import requests
from bs4 import BeautifulSoup
import time

def test_lex_uz():
    """Тест парсинга lex.uz"""
    print("🔍 Тестирование парсинга lex.uz...")
    
    try:
        # Тестируем поиск по конкретному запросу
        search_url = "https://lex.uz/ru/search?q=уголовный+кодекс"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"📡 Отправка запроса на: {search_url}")
        response = requests.get(search_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        print(f"✅ Статус ответа: {response.status_code}")
        print(f"📄 Размер ответа: {len(response.content)} байт")
        
        # Парсим HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Ищем различные элементы
        print("\n🔍 Поиск элементов на странице:")
        
        # Поиск заголовков
        titles = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        print(f"   Заголовки (h1-h6): {len(titles)}")
        
        # Поиск ссылок
        links = soup.find_all('a')
        print(f"   Ссылки: {len(links)}")
        
        # Поиск по классам
        search_results = soup.find_all('div', class_='search-result')
        print(f"   div.search-result: {len(search_results)}")
        
        results = soup.find_all('div', class_='result')
        print(f"   div.result: {len(results)}")
        
        # Поиск по ID
        content_div = soup.find('div', id='content')
        print(f"   div#content: {'Найден' if content_div else 'Не найден'}")
        
        main_div = soup.find('div', id='main')
        print(f"   div#main: {'Найден' if main_div else 'Не найден'}")
        
        # Показываем первые несколько ссылок
        print(f"\n🔗 Первые 5 ссылок:")
        for i, link in enumerate(links[:5]):
            href = link.get('href', '')
            text = link.get_text(strip=True)[:50]
            print(f"   {i+1}. {text} -> {href}")
        
        # Показываем структуру страницы
        print(f"\n📋 Структура страницы:")
        for tag in soup.find_all(['div', 'section', 'article'])[:10]:
            class_attr = tag.get('class', [])
            id_attr = tag.get('id', '')
            if class_attr or id_attr:
                print(f"   <{tag.name} class='{class_attr}' id='{id_attr}'>")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании lex.uz: {e}")
        return False

def test_norma_uz():
    """Тест парсинга norma.uz"""
    print("\n🔍 Тестирование парсинга norma.uz...")
    
    try:
        # Тестируем поиск
        search_url = "https://norma.uz/search?q=уголовный+кодекс"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"📡 Отправка запроса на: {search_url}")
        response = requests.get(search_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        print(f"✅ Статус ответа: {response.status_code}")
        print(f"📄 Размер ответа: {len(response.content)} байт")
        
        # Парсим HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Ищем различные элементы
        print("\n🔍 Поиск элементов на странице:")
        
        # Поиск заголовков
        titles = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        print(f"   Заголовки (h1-h6): {len(titles)}")
        
        # Поиск ссылок
        links = soup.find_all('a')
        print(f"   Ссылки: {len(links)}")
        
        # Поиск по классам
        search_results = soup.find_all('div', class_='search-result')
        print(f"   div.search-result: {len(search_results)}")
        
        results = soup.find_all('div', class_='result')
        print(f"   div.result: {len(results)}")
        
        # Поиск по ID
        content_div = soup.find('div', id='content')
        print(f"   div#content: {'Найден' if content_div else 'Не найден'}")
        
        main_div = soup.find('div', id='main')
        print(f"   div#main: {'Найден' if main_div else 'Не найден'}")
        
        # Показываем первые несколько ссылок
        print(f"\n🔗 Первые 5 ссылок:")
        for i, link in enumerate(links[:5]):
            href = link.get('href', '')
            text = link.get_text(strip=True)[:50]
            print(f"   {i+1}. {text} -> {href}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании norma.uz: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("🚀 Тестирование парсинга сайтов")
    print("=" * 50)
    
    # Тестируем lex.uz
    lex_success = test_lex_uz()
    
    # Пауза между запросами
    time.sleep(2)
    
    # Тестируем norma.uz
    norma_success = test_norma_uz()
    
    print("\n" + "=" * 50)
    print("📊 Результаты тестирования:")
    print(f"   lex.uz: {'✅ Успешно' if lex_success else '❌ Ошибка'}")
    print(f"   norma.uz: {'✅ Успешно' if norma_success else '❌ Ошибка'}")
    
    if lex_success and norma_success:
        print("\n🎯 Оба сайта доступны для парсинга!")
    else:
        print("\n⚠️ Есть проблемы с парсингом некоторых сайтов")

if __name__ == "__main__":
    main()
