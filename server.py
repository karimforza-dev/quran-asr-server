import os
import re
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from rapidfuzz import fuzz
from transformers import pipeline


# التشكيل والتطويل وعلامات الوقف
MARKS = re.compile(r"[\u064B-\u065F\u0640\u06D6-\u06ED]")


def normalize_arabic(text):
    # 9. تحويل إلى أحرف صغيرة (احتياط عند وجود أحرف لاتينية)
    text = text.lower()

    # 1. إزالة التشكيل: الفتحة، الضمة، الكسرة، السكون، الشدة،
    #    التنوين بكل أشكاله ( MARKS )، وتحويل الألف الخنجرية U+0670 إلى ا
    text = MARKS.sub("", text)
    text = text.replace("\u0670", "ا")

    # 2. إزالة علامات الترقيم والوقف
    punctuation = "،؛؟.!«»()[]/\\-_\"'"
    text = re.sub("[" + re.escape(punctuation) + "]", "", text)

    # 3. توحيد الهمزات أ إ آ ٱ ا → ا
    for char in "أإآٱا":
        text = text.replace(char, "ا")

    # 4. توحيد الواو المهموزة ؤ → و
    text = text.replace("ؤ", "و")

    # 5. توحيد الياء المهموزة ئ → ي
    text = text.replace("ئ", "ي")

    # 6. توحيد التاء المربوطة ة → ه
    text = text.replace("ة", "ه")

    # 7. توحيد الألف المقصورة ى → ي
    text = text.replace("ى", "ي")

    # 8. إزالة المسافات الزائدة وتحويل المتعددة إلى مسافة واحدة
    words = text.split()

    # 10. إزالة واو/باء/فاء العطف الزائدة في بداية الكلمات الشائعة جداً،
    #     مع ترك كلمات مثل والدين وواليوم دون تغيير.
    common_after_prefix = {"الحمد", "الله", "العالمين", "الناس"}
    cleaned_words = []
    for word in words:
        if word[:3] in ("وال", "بال", "فال") and word[1:] in common_after_prefix:
            word = word[1:]
        cleaned_words.append(word)

    text = " ".join(cleaned_words)

    # 11. إزالة جميع حروف المد (ا، و، ي) من النص بالكامل
    text = re.sub(r'[اوي]', '', text)

    return text


asr = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


def compare_words(heard_text, ayah_text):
    cleaned_heard = normalize_arabic(heard_text)
    cleaned_ayah = normalize_arabic(ayah_text)

    heard_words = cleaned_heard.split()
    ayah_words = cleaned_ayah.split()

    total = len(ayah_words)
    if total == 0:
        return 0.0, 0, 0, [], []

    # كل كلمة من الآية تُقارَن بأقرب كلمة في النص المسموع
    scores = []
    for ayah_word in ayah_words:
        best_score = max(
            (fuzz.ratio(ayah_word, heard_word) for heard_word in heard_words),
            default=0.0,
        )
        scores.append(best_score)

    # النسبة النهائية = متوسط نسب التشابه لكل كلمة
    percentage = sum(scores) / total
    matched = sum(1 for score in scores if score == 100.0)

    # الكلمات الزائدة/الناقصة تُستنتج من فرق عدد الكلمات
    extra = heard_words[total:] if len(heard_words) > total else []
    missing = ayah_words[len(heard_words):] if len(ayah_words) > len(heard_words) else []

    return percentage, matched, total, missing, extra


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check")
def check(audio: UploadFile = File(...), ayah: str = Form(...)):
    print(repr(ayah))
    with tempfile.TemporaryDirectory() as tmpdir:
        source = os.path.join(tmpdir, "audio.m4a")
        wav_path = os.path.join(tmpdir, "audio.wav")

        with open(source, "wb") as output:
            shutil.copyfileobj(audio.file, output)

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", source, "-vn",
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        result = asr(wav_path, generate_kwargs={"language": "arabic"})
        heard_text = result["text"]

    percentage, matched, total, missing, extra = compare_words(heard_text, ayah)
    return {
        "heard": heard_text,
        "matched": matched,
        "total": total,
        "percent": percentage,
        "missing": missing,
        "extra": extra,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
