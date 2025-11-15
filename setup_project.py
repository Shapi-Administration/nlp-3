# setup_project.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import json

def create_project_structure():
    """Создание структуры проекта"""
    directories = [
        'models/classical_ml',
        'models/neural_networks', 
        'models/transformers',
        'data',
        'src'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Создана папка: {directory}")

def create_demo_models():
    """Создание демо-моделей"""
    # Демо-данные для обучения
    texts = [
        "компания показала рост прибыли рекордный успех развитие",
        "кризис проблемы убытки сокращение конфликт потери",
        "отчет данные анализ статистика информация исследование",
        "инвестиции развитие перспективы будущее возможности",
        "конфликт проблемы трудности сложности вызовы",
        "новости события обзор мониторинг наблюдение",
        "успех достижение победа награда результат",
        "проблемы ошибки недочеты недостатки сложности",
        "стабильность устойчивость надежность постоянство",
        "риски угрозы опасности вызовы проблемы"
    ]
    
    labels = [2, 0, 1, 2, 0, 1, 2, 0, 1, 0]  # 0: negative, 1: neutral, 2: positive
    
    # Создаем TF-IDF векторизатор
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    # Модели для многоклассовой классификации
    models = {
        'logistic_regression': LogisticRegression(),
        'random_forest': RandomForestClassifier(),
        'svm': SVC(probability=True),
        'naive_bayes': MultinomialNB()
    }
    
    for name, model in models.items():
        model.fit(X, labels)
        
        model_data = {
            'model': model,
            'vectorizer': vectorizer,
            'type': 'multiclass'
        }
        
        joblib.dump(model_data, f'models/classical_ml/{name}.joblib')
        print(f"✅ Создана модель: {name}")
    
    # Бинарная модель
    binary_labels = [1 if label == 2 else 0 for label in labels]  # positive vs not positive
    binary_model = LogisticRegression()
    binary_model.fit(X, binary_labels)
    
    binary_data = {
        'model': binary_model,
        'vectorizer': vectorizer, 
        'type': 'binary'
    }
    
    joblib.dump(binary_data, 'models/classical_ml/logistic_binary.joblib')
    print("✅ Создана бинарная модель")

def create_demo_data():
    """Создание демо-данных"""
    # Демо тестовые данные
    test_data = pd.DataFrame({
        'text': [
            "Компания показала отличные результаты и рост",
            "Кризис привел к большим потерям",
            "Состоялось заседание совета директоров",
            "Прибыль компании значительно выросла",
            "Проблемы с поставками вызвали задержки",
            "Инновационные разработки принесли успех",
            "Сокращение штата неизбежно",
            "Анализ рынка показал положительную динамику"
        ],
        'label': ['positive', 'negative', 'neutral', 'positive', 
                 'negative', 'positive', 'negative', 'positive']
    })
    
    test_data.to_csv('data/test.csv', index=False, encoding='utf-8')
    print("✅ Созданы тестовые данные")
    
    # Демо данные для анализа ошибок
    error_data = {
        "confusion_matrices": {
            "logistic_regression": [[45, 8, 2], [5, 38, 7], [3, 6, 41]],
            "random_forest": [[42, 10, 3], [6, 36, 8], [4, 7, 39]],
            "rubert_tiny": [[48, 5, 2], [3, 42, 5], [2, 4, 44]]
        },
        "error_examples": [
            {
                "text": "Компания показала рост, но качество оставляет желать лучшего",
                "true_label": "neutral",
                "predictions": {
                    "logistic_regression": "positive",
                    "random_forest": "positive",
                    "rubert_tiny": "neutral"
                }
            }
        ],
        "common_errors": {
            "mixed_sentiment": 45,
            "irony_sarcasm": 23, 
            "context_dependent": 34,
            "rare_words": 18
        }
    }
    
    with open('data/error_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(error_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Созданы данные для анализа ошибок")

if __name__ == "__main__":
    print("🚀 Создаем структуру проекта...")
    create_project_structure()
    
    print("\n🤖 Создаем демо-модели...")
    create_demo_models()
    
    print("\n📊 Создаем демо-данные...")
    create_demo_data()
    
    print("\n🎉 Проект успешно создан!")
    print("\n📋 Для запуска выполните:")
    print("   streamlit run web_interface.py")