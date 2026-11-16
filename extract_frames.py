"""
استخراج فريمات المنتجات النظيفة (بدون أيدي، واضحة، وغير مكررة) من الفيديوهات.

Extract clean product frames (no hands, sharp, non-duplicate) from videos.

Why not HSV skin detection: the products here are beige/brown cabinets whose
colour falls inside the classic skin range, so 15-90% of every frame was
flagged as "skin" and nothing passed. Hands are detected with MediaPipe's
HandLandmarker instead.

Usage:
    python3 extract_frames.py [input_dir] [output_dir]
"""
import glob
import os
import sys
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
SEG_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
           "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def model_file(url):
    path = os.path.join(MODELS_DIR, url.rsplit("/", 1)[-1])
    if not os.path.exists(path):
        os.makedirs(MODELS_DIR, exist_ok=True)
        print(f"-> تحميل النموذج: {os.path.basename(path)}")
        urllib.request.urlretrieve(url, path)
    return path


def load_hand_detector():
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_file(MODEL_URL)),
        num_hands=2,
        min_hand_detection_confidence=0.3,  # منخفض عمداً: الأفضل رفض فريم سليم من قبول فريم فيه يد
        min_hand_presence_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(opts)


def load_body_segmenter():
    """يصنّف كل بكسل: خلفية/شعر/جلد الجسم/جلد الوجه/ملابس — يلتقط الذراع والكُم حتى لو لم تُكتشف اليد."""
    opts = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_file(SEG_URL)),
        output_confidence_masks=True, output_category_mask=False)
    return vision.ImageSegmenter.create_from_options(opts)


def person_scores(detector, segmenter, frame):
    """يعيد (هل اكتُشفت يد، نسبة بكسلات الجسم بثقة>0.5، نسبة بكسلات الجسم بثقة>0.3)."""
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hand = bool(detector.detect(image).hand_landmarks)
    masks = segmenter.segment(image).confidence_masks
    body = np.maximum.reduce([masks[k].numpy_view() for k in (2, 3, 4)])  # body-skin, face-skin, clothes
    return hand, float((body > 0.5).mean()), float((body > 0.3).mean())


def sharpness(frame):
    return cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def extract_clean_products_from_folder(input_dir=".", output_dir="clean_product_images",
                                       step_seconds=0.5, neighbor_seconds=(0.25, 0.5), strict_body_limit=0.0003, loose_body_limit=0.002,
                                       min_sharpness=40, relative_sharpness=0.8,
                                       min_gap_seconds=2.0):
    os.makedirs(output_dir, exist_ok=True)

    video_files = []
    for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.mkv"):
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    video_files = sorted(set(video_files))

    if not video_files:
        print(f"-> لم يتم العثور على أي فيديوهات في المجلد: {os.path.abspath(input_dir)}")
        return

    print(f"-> تم العثور على {len(video_files)} فيديو. جاري المعالجة...")
    detector = load_hand_detector()
    segmenter = load_body_segmenter()
    total_saved = 0

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        base_name = os.path.splitext(video_name)[0]
        print(f"جاري معالجة: {video_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  - تعذر قراءة الفيديو: {video_name}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            continue

        step = max(1, int(round(fps * step_seconds)))
        offsets = sorted({max(1, int(round(fps * t))) for t in neighbor_seconds})
        candidates = list(range(0, len(frames), step))

        # 1. الحدة: عتبة نسبية لكل فيديو (لأن جودة التصوير تختلف بين الفيديوهات)
        sharp = {i: sharpness(frames[i]) for i in candidates}
        limit = max(min_sharpness, relative_sharpness * float(np.median(list(sharp.values()))))

        # 2. الأيدي/الذراع: الفريم والفريمات المجاورة (±0.25 و ±0.5 ثانية) يجب أن تكون خالية
        cache = {}

        def scores(i):
            if i not in cache:
                cache[i] = person_scores(detector, segmenter, frames[i])
            return cache[i]

        def hand_at(i, strict):
            hand, loose_frac, strict_frac = scores(i)
            if strict:  # الفريم نفسه: أي أثر صغير لأصابع يرفضه
                return hand or strict_frac > strict_body_limit
            return hand or loose_frac > loose_body_limit  # الفريمات المجاورة: فحص أخف

        clean = []
        stats = {"blurry": 0, "hand": 0}
        for i in candidates:
            if sharp[i] < limit:
                stats["blurry"] += 1
                continue
            neighbors = [i + d * o for o in offsets for d in (-1, 1) if 0 <= i + d * o < len(frames)]
            if hand_at(i, strict=True) or any(hand_at(j, strict=False) for j in neighbors):
                stats["hand"] += 1
                continue
            clean.append(i)

        # 3. إزالة التكرار: الفريمات النظيفة المتتالية هي نفس اللقطة تقريباً،
        #    نختار الأوضح من كل مقطع متصل، وبحد أقصى صورة واحدة لكل min_gap_seconds
        runs, run = [], []
        for i in clean:
            if run and i - run[-1] > step:
                runs.append(run)
                run = []
            run.append(i)
        if run:
            runs.append(run)
        gap = int(fps * min_gap_seconds)
        kept = []
        for run in runs:
            for start in range(run[0], run[-1] + 1, gap):
                chunk = [i for i in run if start <= i < start + gap]
                if chunk:
                    kept.append(max(chunk, key=lambda k: sharp[k]))

        for n, i in enumerate(kept):
            t = i / fps
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_{n:02d}_t{t:05.1f}s.png"), frames[i])
        total_saved += len(kept)
        print(f"  - فحص {len(candidates)} فريم: {stats['blurry']} غير واضح، {stats['hand']} فيه يد، "
              f"{len(clean) - len(kept)} مكرر -> تم حفظ {len(kept)} صورة.")

    detector.close()
    segmenter.close()
    print(f"\n-> اكتملت العملية. إجمالي الصور المستخرجة: {total_saved} في مجلد: {output_dir}")


if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "clean_product_images"
    extract_clean_products_from_folder(input_dir=in_dir, output_dir=out_dir)
