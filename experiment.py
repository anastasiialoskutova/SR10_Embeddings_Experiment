import time
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Корпус документів (база знань e-commerce, рекомендацій та фінансів, UA + EN)
CORPUS = [
    {"id": "doc_01", "text": "Колаборативна фільтрація рекомендує товари на основі історії покупок схожих користувачів."},
    {"id": "doc_02", "text": "Контентна фільтрація аналізує опис, категорію та характеристики товару для пошуку аналогів."},
    {"id": "doc_03", "text": "Проблема холодного старту виникає для нових користувачів або товарів без історії взаємодій."},
    {"id": "doc_04", "text": "Кешбек нараховується у розмірі 5% на електроніку при оплаті преміальною карткою."},
    {"id": "doc_05", "text": "Повернення коштів за скасоване замовлення здійснюється на банківську картку протягом 3 робочих днів."},
    {"id": "doc_06", "text": "Динамічне ціноутворення змінює вартість товару залежно від попиту, залишків на складі та часу доби."},
    {"id": "doc_07", "text": "RFM-аналіз сегментує клієнтів за давністю покупки (Recency), частотою (Frequency) та сумою витрат (Monetary)."},
    {"id": "doc_08", "text": "Антифрод-система блокує підозрілі транзакції при раптовій зміні геолокації або пристрою платника."},
    {"id": "doc_09", "text": "Векторні бази даних використовують індекс HNSW для швидкого наближеного пошуку найближчих сусідів."},
    {"id": "doc_10", "text": "Косинусна подібність вимірює кут між двома нормалізованими векторами ембедінгів."},
    {"id": "doc_11", "text": "Гібридний пошук поєднує лексичний алгоритм BM25 та семантичний векторний пошук для підвищення повноти."},
    {"id": "doc_12", "text": "Крос-енкодер виконує повторне ранжування (reranking) топ-кандидатів для максимальної точності відповіді."},
    {"id": "doc_13", "text": "Програма лояльності дозволяє накопичувати бонусні бали та списувати їх до 50% вартості чека."},
    {"id": "doc_14", "text": "Безкоштовна доставка діє для замовлень на суму від 1500 гривень по всій території України."},
    {"id": "doc_15", "text": "Розстрочка без переплат доступна на термін до 6 місяців для товарів дорожчих за 3000 грн."},
    {"id": "doc_16", "text": "Collaborative filtering recommends items based on the purchase history of similar users."},
    {"id": "doc_17", "text": "The cold start problem occurs when new users or products have no historical interaction data."},
    {"id": "doc_18", "text": "Fraud detection systems flag anomalous payments triggered by sudden IP address or device changes."},
    {"id": "doc_19", "text": "Hybrid search combines sparse lexical BM25 retrieval with dense semantic vector embeddings."},
    {"id": "doc_20", "text": "A/B testing evaluates recommendation click-through rate (CTR) and conversion uplift in production."},
    {"id": "doc_21", "text": "Клієнтський відтік (churn rate) прогнозується за зниженням частоти візитів у мобільний застосунок."},
    {"id": "doc_22", "text": "Апселінг (upselling) пропонує покупцеві дорожчу модель товару з кращими характеристиками."},
    {"id": "doc_23", "text": "Крос-селінг (cross-selling) рекомендує супутні аксесуари, наприклад чохол до смартфона."},
    {"id": "doc_24", "text": "Сесійна рекомендаційна система враховує послідовність кліків користувача у реальному часі."},
    {"id": "doc_25", "text": "Метрика NDCG оцінює якість ранжування рекомендацій з урахуванням позиції релевантного товару."}
]

# 2. 30 тестових запитів із золотою розміткою (UA, EN, перефразування, крос-мовні)
QUERIES = [
    # Прямі та перефразовані запити українською (1-18)
    {"id": "Q01", "lang": "UA", "text": "Як працюють рекомендації на основі схожих покупців?", "gold": "doc_01"},
    {"id": "Q02", "lang": "UA", "text": "Підбір товарів за їхніми атрибутами та категоріями", "gold": "doc_02"},
    {"id": "Q03", "lang": "UA", "text": "Що робити, якщо у нового клієнта ще немає історії замовлень?", "gold": "doc_03"},
    {"id": "Q04", "lang": "UA", "text": "Який відсоток кешбеку повертається за купівлю гаджетів?", "gold": "doc_04"},
    {"id": "Q05", "lang": "UA", "text": "Скільки днів чекати повернення грошей після скасування покупки?", "gold": "doc_05"},
    {"id": "Q06", "lang": "UA", "text": "Автоматична зміна ціни залежно від попиту на товар", "gold": "doc_06"},
    {"id": "Q07", "lang": "UA", "text": "Сегментація клієнтської бази за частотою та сумою чеків", "gold": "doc_07"},
    {"id": "Q08", "lang": "UA", "text": "Захист від шахрайських оплат при зміні пристрою", "gold": "doc_08"},
    {"id": "Q09", "lang": "UA", "text": "Який графовий індекс використовують для швидкого пошуку векторів?", "gold": "doc_09"},
    {"id": "Q10", "lang": "UA", "text": "Як обчислюється схожість між двома нормалізованими ембедінгами?", "gold": "doc_10"},
    {"id": "Q11", "lang": "UA", "text": "Поєднання пошуку за ключовими словами BM25 та векторів", "gold": "doc_11"},
    {"id": "Q12", "lang": "UA", "text": "Переранжування знайдених документів крос-енкодером", "gold": "doc_12"},
    {"id": "Q13", "lang": "UA", "text": "Як витратити накопичені бонуси на оплату частини замовлення?", "gold": "doc_13"},
    {"id": "Q14", "lang": "UA", "text": "Мінімальна сума кошика для безкоштовної доставки по Україні", "gold": "doc_14"},
    {"id": "Q15", "lang": "UA", "text": "Оплата частинами без відсотків на пів року", "gold": "doc_15"},
    {"id": "Q16", "lang": "UA", "text": "Як передбачити, що користувач перестане користуватися додатком?", "gold": "doc_21"},
    {"id": "Q17", "lang": "UA", "text": "Пропозиція дорожчої версії продукту під час вибору", "gold": "doc_22"},
    {"id": "Q18", "lang": "UA", "text": "Рекомендація додаткових аксесуарів у кошик", "gold": "doc_23"},
    # Перефразування складними формулюваннями UA (19-22)
    {"id": "Q19", "lang": "UA", "text": "Рекомендації в реальному часі на базі поточних дій на сайті", "gold": "doc_24"},
    {"id": "Q20", "lang": "UA", "text": "Оцінка порядку видачі рекомендацій із вагою позиції", "gold": "doc_25"},
    {"id": "Q21", "lang": "UA", "text": "Алгоритм схожих юзерів для e-commerce", "gold": "doc_01"},
    {"id": "Q22", "lang": "UA", "text": "Бонусна система для постійних покупців", "gold": "doc_13"},
    # Англомовні та крос-мовні запити (23-30)
    {"id": "Q23", "lang": "EN", "text": "How does user-based collaborative filtering work?", "gold": "doc_16"},
    {"id": "Q24", "lang": "EN", "text": "Handling recommendations when a new user has zero interactions", "gold": "doc_17"},
    {"id": "Q25", "lang": "EN", "text": "Detecting suspicious transactions from an unrecognized device", "gold": "doc_18"},
    {"id": "Q26", "lang": "EN", "text": "Combining keyword search BM25 with dense embeddings", "gold": "doc_19"},
    {"id": "Q27", "lang": "EN", "text": "Measuring CTR and conversion lift with online experiments", "gold": "doc_20"},
    {"id": "Q28", "lang": "CROSS", "text": "Як виміряти конверсію та CTR рекомендацій через A/B тестування?", "gold": "doc_20"},
    {"id": "Q29", "lang": "CROSS", "text": "Free shipping threshold in Ukraine for online orders", "gold": "doc_14"},
    {"id": "Q30", "lang": "CROSS", "text": "Segmenting customers by recency, frequency and monetary value", "gold": "doc_07"},
]

MODELS = [
    ("all-MiniLM-L6-v2 (EN, 384d)", "all-MiniLM-L6-v2"),
    ("paraphrase-multilingual-MiniLM-L12-v2 (Multi, 384d)", "paraphrase-multilingual-MiniLM-L12-v2"),
    ("BAAI/bge-base-en-v1.5 (EN-Large, 768d)", "BAAI/bge-base-en-v1.5"),
]

doc_ids = [d["id"] for d in CORPUS]
doc_texts = [d["text"] for d in CORPUS]
query_texts = [q["text"] for q in QUERIES]

all_predictions = {}
summary_rows = []

for label, model_name in MODELS:
    print(f"\nЗавантаження та тестування моделі: {label}...")
    model = SentenceTransformer(model_name)
    
    # Індексація корпусу (з L2-нормалізацією)
    t_idx0 = time.perf_counter()
    doc_emb = model.encode(doc_texts, normalize_embeddings=True)
    t_idx1 = time.perf_counter()
    
    # Розмір індексу в пам'яті (байти -> КБ)
    index_size_kb = doc_emb.nbytes / 1024.0
    
    # Прогрів та замір затримки кодування запитів
    _ = model.encode(query_texts[:2], normalize_embeddings=True)
    t_q0 = time.perf_counter()
    q_emb = model.encode(query_texts, normalize_embeddings=True)
    t_q1 = time.perf_counter()
    
    avg_latency_ms = ((t_q1 - t_q0) / len(QUERIES)) * 1000.0
    
    # Обчислення косинусної подібності (через скалярний добуток нормалізованих векторів)
    scores = np.matmul(q_emb, doc_emb.T)
    
    hits_at_5 = 0
    hits_ua = 0
    total_ua = sum(1 for q in QUERIES if q["lang"] == "UA")
    mrr_sum = 0.0
    preds_for_model = {}
    
    for i, q in enumerate(QUERIES):
        ranked_indices = np.argsort(scores[i])[::-1]
        ranked_ids = [doc_ids[idx] for idx in ranked_indices]
        top5 = ranked_ids[:5]
        preds_for_model[q["id"]] = (top5[0], top5, round(float(scores[i][ranked_indices[0]]), 3))
        
        if q["gold"] in top5:
            hits_at_5 += 1
            if q["lang"] == "UA":
                hits_ua += 1
        
        rank = ranked_ids.index(q["gold"]) + 1
        mrr_sum += 1.0 / rank
        
    recall_5 = (hits_at_5 / len(QUERIES)) * 100.0
    recall_5_ua = (hits_ua / total_ua) * 100.0
    mrr = mrr_sum / len(QUERIES)
    
    all_predictions[label] = preds_for_model
    summary_rows.append((label, recall_5, recall_5_ua, mrr, avg_latency_ms, index_size_kb))

print("\n" + "="*105)
print(f"{'Модель':<52} | {'Recall@5':<9} | {'R@5 (UA)':<9} | {'MRR':<6} | {'Latency (мс)':<12} | {'Індекс (КБ)'}")
print("="*105)
for row in summary_rows:
    print(f"{row[0]:<52} | {row[1]:>7.1f}% | {row[2]:>7.1f}% | {row[3]:>6.3f} | {row[4]:>12.2f} | {row[5]:>10.2f}")
print("="*105)

print("\n--- АНАЛІЗ РОЗБІЖНОСТЕЙ МІЖ МОДЕЛЯМИ (Топ-1 прогноз) ---")
m1_name, m2_name, m3_name = [m[0] for m in MODELS]
diff_count = 0
for q in QUERIES:
    p1 = all_predictions[m1_name][q["id"]]
    p2 = all_predictions[m2_name][q["id"]]
    p3 = all_predictions[m3_name][q["id"]]
    # Виводимо запити, де моделі дали різні топ-1 або помилилися
    if len({p1[0], p2[0], p3[0]}) > 1 or p1[0] != q["gold"]:
        diff_count += 1
        print(f"\n[{q['id']} | {q['lang']}] Запит: \"{q['text']}\" (Gold: {q['gold']})")
        print(f"  - MiniLM-L6 (EN):      Top1={p1[0]} (score={p1[2]}), Top5={p1[1]}")
        print(f"  - Multilingual-L12:    Top1={p2[0]} (score={p2[2]}), Top5={p2[1]}")
        print(f"  - BGE-base-en (768d):  Top1={p3[0]} (score={p3[2]}), Top5={p3[1]}")
        if diff_count >= 7:
            break
