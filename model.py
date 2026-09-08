
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib


df = pd.read_csv("emails.csv")

df = df.drop_duplicates(subset="text").reset_index(drop=True)



STOPWORDS = """و در به از که را با این است می برای آن یک خود تا هم های بود شد
نیز ما من شما او اگر پس یا هر چند دارد باشد باید اینکه همان طور دیگر حتی
توسط مورد قرار بعد قبل روی چون بدون ولی اما آیا حال تمام داده شده کرد کند""".split()


SPAM_WORDS = """barande bordid jayeze jaize tabrik takhfif haraj forsat foori rayegan
sabtenam mahdood hadiye sood vam toman milyon click klik win winner prize bonus
jackpot lottery reward gift cash free discount promo coupon vip urgent guaranteed
congratulations congrats crypto bitcoin forex invest profit claim verify offer
limited exclusive""".split()

SAFE_WORDS = """ok okay hi hello bye thanks thank please sorry yes no dear best
regards meeting zoom skype teams slack google gmail email mail file files pdf doc
docx excel word ppt png jpg zip csv report project team deadline update version
test server api admin user github drive folder invoice attachment agenda minutes
review draft final budget sql python java html css js ui ux qa hr it ceo cto pm
cc fw re fyi asap done call""".split()


SPAM_LATIN = r"(?i)\b(" + "|".join(SPAM_WORDS) + r")\b"
SAFE_LATIN = r"(?i)\b(" + "|".join(SAFE_WORDS) + r")\b"


PRIZE_EMOJI = "[\U0001F380-\U0001F384\U0001F389\U0001F38A\U0001F3C6\U0001F451\U0001F48E\U0001F4B0-\U0001F4B8\U0001F911\U0001F947\u2705\u2714\u2728\u26A1\u2757\U0001F525\U0001F6A8\U0001F4E2\U0001F514\u23F0]+"
ANY_EMOJI = "[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F]+"


def cleaning(text_series):
    s = text_series.fillna("").astype(str)

    for i, d in enumerate("۰۱۲۳۴۵۶۷۸۹"):
        s = s.str.replace(d, str(i))
    for i, d in enumerate("٠١٢٣٤٥٦٧٨٩"):
        s = s.str.replace(d, str(i))

    s = s.str.replace(r"[\w\.-]+@[\w\.-]+\.\w+", " نشانیمیل ", regex=True)
    s = s.str.replace(r"@[A-Za-z_]\w{3,}|t\.me/\S+", " ایدیتلگرام ", regex=True)
    s = s.str.replace(
        r"http\S+|www\.\S+|\S*\.(?:com|net|org|info|xyz|ir|co|me|dev|io|shop|li)\b\S*", " لینک ", regex=True
    )
    s = s.str.replace(r"\b0?9\d{9}\b", " شمارهتلفن ", regex=True)
    s = s.str.replace(r"\b(?:\d{4}[- ]?){3}\d{4}\b", " شمارهکارت ", regex=True)
    s = s.str.replace(r"\bIR\d{20,24}\b", " شمارهشبا ", regex=True)
    s = s.str.replace(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", " آیپی ", regex=True)
    s = s.str.replace(
        r"\d[\d,\.٬]*\s*(?:تومان|ریال|میلیون|میلیارد|هزار|دلار|یورو)", " مبلغپول ", regex=True
    )
    s = s.str.replace(r"\d+\s*(?:%|٪|درصد)", " درصدسود ", regex=True)
    s = s.str.replace(r"\d{5,}", " عددبلند ", regex=True)

    s = s.str.replace(r"!+", " علامتتعجب ", regex=True)
    s = s.str.replace(r"[؟?]+", " علامتسوال ", regex=True)

    s = s.str.replace(PRIZE_EMOJI, " ایموجیجایزه ", regex=True)
    s = s.str.replace(ANY_EMOJI, " ایموجی ", regex=True)

    s = s.str.replace(SPAM_LATIN, " لاتیناسپم ", regex=True)
    s = s.str.replace(SAFE_LATIN, " لاتینرایج ", regex=True)
    s = s.str.replace(r"[A-Za-z]{2,}", " حرفلاتین ", regex=True)

    s = s.str.replace("ي", "ی").str.replace("ك", "ک")
    s = s.str.replace("أ", "ا").str.replace("إ", "ا")
    s = s.str.replace("آ", "ا").str.replace("ة", "ه")
    s = s.str.replace("\u200c", "")
    s = s.str.replace(r"\b(ن?می) (?=\S)", r"\1", regex=True)

    s = s.str.replace(r"[^\u0621-\u06cc ]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    s = s.str.replace(
        r"(?<![\u0621-\u06cc])([\u0621-\u06cc]{1,2}(?: [\u0621-\u06cc]){2,})(?![\u0621-\u06cc])",
        lambda m: " حروفجداشده " + m.group(1).replace(" ", "") + " ",
        regex=True,
    )

    s = s.str.replace("\u0640", "")
    s = s.str.replace(r"([او])\1+", r"\1", regex=True)
    s = s.str.replace(r"(.)\1{2,}", r"\1\1", regex=True)

    s = s.str.replace(r"(\S{3,})(?<!اژد)هایی\b", r"\1", regex=True)
    s = s.str.replace(r"(\S{3,})(?<!اژد)های\b", r"\1", regex=True)
    s = s.str.replace(r"(\S{3,})(?<!اژد)ها\b", r"\1", regex=True)

    s = s.str.replace(r"(?<!\S)(\S+)(?: \1){2,}(?!\S)", r"\1 \1", regex=True)

    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s



df["clean_text"] = cleaning(df["text"])


X = df["clean_text"]
y = df["label"].map({"spam": 1, "normal": 0})



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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


model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_v, y_train)


THRESHOLD = 0.45

def spam_probability(matrix):
    return model.predict_proba(matrix)[:, 1]


y_pred = pd.Series((spam_probability(X_test_v) >= THRESHOLD).astype(int), index=X_test.index)
acc_train = accuracy_score(y_train, (spam_probability(X_train_v) >= THRESHOLD).astype(int))
acc_test = accuracy_score(y_test, y_pred)

box("دقت مدل")
print(f"   ✦ آستانه تصمیم اسپم   : {THRESHOLD}")
print(f"   ✦ دقت روی داده آموزش : {acc_train:.4f}")
print(f"   ✦ دقت روی داده آزمون  : {acc_test:.4f}")

print(f"   ✦ فاصله ی دو عدد  : {acc_train - acc_test:.4f}  ")


cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["واقعی: عادی", "واقعی: اسپم"],
    columns=["پیش‌بینی: عادی", "پیش‌بینی: اسپم"],
)


box("ماتریس درهم‌ریختگی")

print(cm_df)



tn, fp, fn, tp = cm.ravel()
box("معنی چهار خانه")
print(f"   ✦ TN = {tn:<4} ایمیل عادی که درست عادی تشخیص شد")
print(f"   ✦ FP = {fp:<4} ایمیل عادی که اشتباهاً اسپم شد (پرهزینه‌ترین خطا)")
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
box("گزارش طبقه‌بندی")
print(rep_df.round(4))


if "source" in df.columns:
    src_test = df.loc[X_test.index, "source"]
    box("دقت جداگانه هر منبع")
    for src in src_test.unique():
        mask = src_test == src
        print(f"   ✦ {src:10s} n = {mask.sum():5d}   دقت = {accuracy_score(y_test[mask], y_pred[mask]):.4f}")


joblib.dump(
    {"vectorizer": vectorizer, "model": model, "threshold": THRESHOLD},
    "spam_model.joblib",
)


def predict(texts):
    clean = cleaning(pd.Series(list(texts)))
    probs = spam_probability(vectorizer.transform(clean))
    labels = ["spam" if p >= THRESHOLD else "normal" for p in probs]
    return pd.DataFrame({"label": labels, "spam_probability": probs.round(4)})


box("ذخیرهٔ مدل")
print("   ✦ مدل و بردارساز در فایل spam_model.joblib ذخیره شدند")

print("\n" + "━" * 40 + " ✦")
