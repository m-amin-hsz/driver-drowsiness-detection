# 🚗 Driver Drowsiness Detection System (سیستم تشخیص خواب‌آلودگی راننده)

یک سیستم بینایی ماشین و هوش مصنوعی برای پایش بلادرنگ (Real-Time) هوشیاری راننده با استفاده از نقاط کلیدی چهره (Face Mesh Landmarks) و یادگیری ماشین.

---

## 👥 تفکیک وظایف تیم (نفر اول و نفر دوم)

پروژه به دو بخش کاملاً مجزا و ماژولار تقسیم شده است:

| نقش | مسئولیت‌ها | ابزارها و خروجی‌ها | وضعیت |
| :--- | :--- | :--- | :---: |
| **👤 نفر اول (شما)**<br>Computer Vision & Data Pipeline | ۱. انتخاب، دانلود و پاکسازی دیتاست<br>۲. پیش‌پردازش و تفکیک (Train/Val/Test Split)<br>۳. تشخیص چهره و لندمارک‌های چشم و دهان<br>۴. محاسبه معیارهای EAR, MAR, PERCLOS, Blink Rate<br>۵. تولید دیتاست فیچرهای ساختاریافته (CSV) | MediaPipe FaceLandmarker, OpenCV, Pandas<br>**خروجی:** فایل‌های CSV فیچرها در `data/features/` | ✅ **تکمیل شد (مرحله اول و پایپ‌لاین آماده)** |
| **👤 نفر دوم (هم‌تیمی)**<br>Machine Learning & Application | ۱. انتخاب و مقایسه الگوریتم‌های ML (SVM, Random Forest, XGBoost)<br>۲. آموزش و اعتبارسنجی مدل‌ها روی فیچرهای CSV<br>۳. محاسبه متریک‌ها (Accuracy, Precision, Recall, F1, Confusion Matrix)<br>۴. پیش‌بینی بلادرنگ (Real-Time Inference)<br>۵. پیاده‌سازی سیستم هشدار صوتی و تصویری (Alarm HUD) | Scikit-Learn, PyTorch/XGBoost, Pygame/Beep, OpenCV GUI<br>**خروجی:** مدل ذخیره‌شده (`model.pkl`) و برنامه نهایی هشدار | ⏳ آماده ورود داده‌های مرحله ۱ |

---

## 📁 ساختار پروژه

```text
binaii/
├── data/
│   ├── raw/                  # تصاویر و ویدیوهای خام (دانلود شده یا ضبط شده)
│   │   └── sample/           # نمونه‌های اولیه تست (awake / drowsy)
│   ├── splits/               # مانیفست‌های تفکیک داده
│   │   ├── train_manifest.csv
│   │   ├── val_manifest.csv
│   │   └── test_manifest.csv
│   └── features/             # فایل‌های خروجی فیچرها (تحویل به نفر دوم)
│       ├── features_train.csv
│       ├── features_val.csv
│       └── features_test.csv
├── models/
│   └── face_landmarker.task  # مدل وزن‌های MediaPipe Face Mesh
├── src/
│   ├── __init__.py
│   ├── dataset_downloader.py # معرفی و دانلود دیتاست‌های استاندارد
│   ├── dataset_prep.py       # اسکن، پاکسازی و تقسیم به Train/Val/Test
│   ├── data_collector.py     # ابزار ضبط نمونه از وبکم توسط کاربر
│   ├── face_mesh.py          # ماژول استخراج چهره و لندمارک‌های MediaPipe
│   ├── feature_extraction.py # فرمول‌های ریاضی EAR, MAR, PERCLOS, Blink Rate
│   └── pipeline.py           # پایپ‌لاین کامل استخراج فیچرها و حالت زنده (Live)
├── requirements.txt
└── README.md
```

---

## 📐 فرمول‌های فیچرها (Feature Engineering)

### ۱. Eye Aspect Ratio (EAR)
برای تشخیص وضعیت باز یا بسته بودن چشم‌ها بر اساس ۶ نقطه لندمارک:
$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$
- چشم باز: $EAR \approx 0.30 - 0.40$
- چشم بسته: $EAR < 0.21$

### ۲. Mouth Aspect Ratio (MAR)
برای تشخیص خمیازه (Yawning):
$$MAR = \frac{||p_{top} - p_{bottom}||}{||p_{left} - p_{right}||}$$
- حالت عادی / صحبت: $MAR < 0.50$
- خمیازه: $MAR > 0.65$

### ۳. PERCLOS (Percentage of Eye Closure)
مهم‌ترین شاخص علمی خستگی: درصد فریم‌هایی در یک پنجره زمانی لغزان (مثلاً ۳ ثانیه) که چشم بسته بوده است:
$$PERCLOS = \frac{\text{تعداد فریم‌های چشم بسته در بازه}}{\text{کل فریم‌های بازه}} \times 100$$
- حالت هوشیار: $PERCLOS < 15\%$
- خواب‌آلودگی شدید: $PERCLOS > 35\%$

---

## 🚀 راهنمای اجرا

### ۱. نصب وابستگی‌ها
```bash
pip install -r requirements.txt
```

### ۲. ایجاد دیتای نمونه و مشاهده دیتاست‌های معتبر
```bash
python src/dataset_downloader.py
```

### ۳. آماده‌سازی و ساخت Splitهای داده
```bash
python src/dataset_prep.py --raw_dir data/raw
```

### ۴. استخراج فیچرها به فایل CSV (تحویل به نفر دوم)
```bash
python src/pipeline.py --manifest data/splits/train_manifest.csv --out data/features/features_train.csv
python src/pipeline.py --manifest data/splits/test_manifest.csv --out data/features/features_test.csv
```

### ۵. تست زنده با وبکم (نمایش لندمارک‌ها و مقادیر لحظه‌ای)
```bash
python src/pipeline.py --live
```

### ۶. ضبط داده اختصاصی با وبکم (در صورت نیاز)
```bash
python src/data_collector.py
```
- کلید `a`: ضبط نمونه بیدار (Awake)
- کلید `d`: ضبط نمونه خواب‌آلود (Drowsy)
- کلید `q`: خروج

