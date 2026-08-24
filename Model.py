import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv("email_dataset.csv")

STOPWORDS = """و در به از که را با این است می برای آن یک خود تا هم های بود شد
نیز ما من شما او اگر پس یا هر چند دارد باشد باید اینکه همان طور دیگر حتی
توسط مورد قرار بعد قبل روی چون بدون ولی اما آیا حال تمام داده شده کرد کند""".split()

def cleaning(text_series):
    s = text_series.fillna("").astype(str)

    s = s.str.replace(r"http\S+|www\.\S+", " لینک ", regex=True)
    s = s.str.replace(r"[\w\.-]+@[\w\.-]+", " نشانیمیل ", regex=True)
    s = s.str.replace(r"0?9\d{9}", " شمارهتلفن ", regex=True)
    s = s.str.replace(
        r"\d[\d,\.٬]*\s*(تومان|ریال|میلیون|میلیارد|دلار)", " مبلغپول ", regex=True
    )
    s = s.str.replace(r"\d+\s*(%|درصد)", " درصدسود ", regex=True)
    s = s.str.replace(r"\d{5,}", " عددبلند ", regex=True)

    s = s.str.replace(r"!+", " علامتتعجب ", regex=True)
    s = s.str.replace(r"[A-Za-z]{2,}", " حرفلاتین ", regex=True)

    s = s.str.replace("ي", "ی").str.replace("ك", "ک")
    s = s.str.replace("أ", "ا").str.replace("إ", "ا")
    s = s.str.replace("آ", "ا").str.replace("ة", "ه")
    s = s.str.replace("\u200c", " ")

    s = s.str.replace(r"[^\u0621-\u06cc ]", " ", regex=True)

    s = s.str.replace("\u0640", "")
    s = s.str.replace(r"(.)\1{2,}", r"\1", regex=True)

    s = s.str.replace(r"(\S{3,})های\b", r"\1", regex=True)
    s = s.str.replace(r"(\S{3,})ها\b", r"\1", regex=True)

    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

df["clean_text"] = cleaning(df["text"])

X = df["clean_text"]
y = df["label"].map({"spam": 1, "normal": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    stop_words=STOPWORDS,
)

X_train_v = vectorizer.fit_transform(X_train)
X_test_v = vectorizer.transform(X_test)

def box(text):
    print("\n⌌┄" + "┄" * 42 + "⌍")
    print("  ✦ " + text)
    print("⌎┄" + "┄" * 42 + "⌏")

model = LogisticRegression(max_iter=1000)
model.fit(X_train_v, y_train)

y_pred = pd.Series(model.predict(X_test_v), index=X_test.index)
acc_train = accuracy_score(y_train, model.predict(X_train_v))
acc_test = accuracy_score(y_test, y_pred)

box("دقت مدل")
print(f"   ✦ دقت روی دادهٔ آموزش : {acc_train:.4f}")
print(f"   ✦ دقت روی دادهٔ آزمون  : {acc_test:.4f}")


cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["واقعی: عادی", "واقعی: اسپم"],
    columns=["پیش‌ بینی: عادی", "پیش‌ بینی: اسپم"],
)
box("ماتریس درهم‌ریختگی")
print(cm_df)

tn, fp, fn, tp = cm.ravel()
box("ماتریس در هم ریختگی(2)")
print(f"   ✦ TN = {tn:<4} ایمیل عادی که درست عادی تشخیص شد")
print(f"   ✦ FP = {fp:<4} ایمیل عادی که اشتباهاً اسپم شد (پرهزینه‌ ترین خطا)")
print(f"   ✦ FN = {fn:<4} اسپمی که از فیلتر رد شد")
print(f"   ✦ TP = {tp:<4} اسپمی که درست گرفته شد")

rep = classification_report(
    y_test, y_pred, target_names=["عادی (normal)", "اسپم (spam)"], output_dict=True
)
rep_df = pd.DataFrame(
    [rep["عادی (normal)"], rep["اسپم (spam)"]],
    index=["عادی (normal)", "اسپم (spam)"],
)
rep_df["support"] = rep_df["support"].astype(int)
box("گزارش طبقه‌ بندی")
print(rep_df.round(4))

if "source" in df.columns:
    src_test = df.loc[X_test.index, "source"]
    box("دقت جداگانه هر منبع")
    for src in src_test.unique():
        mask = src_test == src
        print(f"   ✦ {src:10s} n = {mask.sum():5d}   دقت = {accuracy_score(y_test[mask], y_pred[mask]):.4f}")

print("\n" + "━" * 40 + " ✦")
