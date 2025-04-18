
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression

def predict_and_plot():
    try:
        print("🚀 Загружаем данные...")
        df = pd.read_csv("data/sample_data_with_signals.csv")
        print("✅ Данные загружены.")
    except Exception as e:
        print("❌ Ошибка при загрузке данных:", e)
        return

    # Преобразуем сигналы в числовой формат
    df['signal'] = df['signal'].map({'BUY': 1, 'SELL': -1, 'HOLD': 0})

    # Используем close и volume как фичи
    X = df[['close', 'volume']]
    y = df['signal']

    print("🧠 Обучаем модель...")
    model = LogisticRegression()
    model.fit(X, y)
    print(f"🎯 Точность модели: {round(model.score(X, y)*100, 2)}%")

    # Сохраняем модель
    joblib.dump(model, "model.pkl")
    print("✅ Модель сохранена как model.pkl")

    # Предсказание
    y_pred = model.predict(X)

    # Визуализация
    plt.figure(figsize=(14, 6))
    plt.plot(df['timestamp'], df['close'], label='Цена', alpha=0.5)
    plt.scatter(df['timestamp'], df['close'], c=y_pred, cmap='coolwarm', label='Сигналы')
    plt.title("Прогноз сигналов (BUY=1, SELL=-1, HOLD=0)")
    plt.xlabel("Время")
    plt.ylabel("Цена")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("preview.png")
    print("📊 График сохранён как preview.png")

if __name__ == "__main__":
    predict_and_plot()
