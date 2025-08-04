# Настройка Tinkoff Investments API

## Получение API токена

1. Зарегистрируйтесь на [Tinkoff Investments](https://www.tinkoff.ru/invest/)
2. Откройте [личный кабинет](https://www.tinkoff.ru/invest/account/)
3. Перейдите в раздел "Настройки" → "API"
4. Создайте новый токен с правами на чтение данных
5. Скопируйте токен

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Настройка переменной окружения

Установите переменную окружения с вашим токеном:

```bash
export TINKOFF_TOKEN='your_token_here'
```

Для постоянного сохранения добавьте в `~/.bashrc` или `~/.zshrc`:

```bash
echo 'export TINKOFF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

## Тестирование

Запустите тестовый скрипт:

```bash
python test_tinvest_loader.py
```

## Доступные российские фьючерсы

В конфигурации настроены следующие ликвидные фьючерсы:

- **Si** - USD/RUB Futures (доллар/рубль)
- **BR** - Brent Oil Futures (нефть Brent)
- **RI** - RTS Index Futures (индекс РТС)
- **MX** - Moscow Exchange Index Futures (индекс МосБиржи)
- **GD** - Gold Futures (золото)
- **SBRF** - Sberbank Futures (Сбербанк)
- **GAZR** - Gazprom Futures (Газпром)
- **LKOH** - Lukoil Futures (Лукойл)
- **NVTK** - Novatek Futures (Новатэк)
- **ROSN** - Rosneft Futures (Роснефть)

## Использование в коде

```python
from src.data.tinvest_loader import TInvestLoader

# Инициализация
loader = TInvestLoader()

# Загрузка данных
data = loader.download_data(
    symbols=["Si", "BR", "RI"],
    start_date="2023-01-01",
    end_date="2023-12-31"
)

print(data.head())
```

## Ограничения API

- Максимальный период запроса: 1 год
- Лимит запросов: 300 запросов в минуту
- Доступны только дневные данные (OHLCV)
- Данные доступны с 2010 года для большинства инструментов

## Устранение проблем

### Ошибка "TINKOFF_TOKEN environment variable is required"
- Убедитесь, что переменная окружения установлена
- Проверьте правильность токена

### Ошибка "Could not find FIGI for symbol"
- Проверьте правильность тикера
- Некоторые инструменты могут быть недоступны в API

### Ошибка "No candles found"
- Проверьте даты запроса
- Убедитесь, что инструмент торговался в указанный период 