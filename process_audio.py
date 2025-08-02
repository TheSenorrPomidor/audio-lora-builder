#!/usr/bin/env python3
# === Версия ===
print("\n🔢 Версия скрипта process_audio.py 2.0")

import os
import shutil
import json
import subprocess
from pathlib import Path
import time
from collections import Counter

from faster_whisper import WhisperModel
from pyannote.audio import Model, Pipeline
from pyannote.audio.core.io import Audio
from pyannote.core import Segment
import numpy as np
import torch
import torch.nn.functional as F

# === 0. Функции ===
def write_json_v2(segments, base_path, rel_path, you_id, caller_id):
    json_path = base_path.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "file": str(rel_path),
        "segments": [
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "speaker": you_id if seg["cluster"] == 0 else caller_id,
                "text": seg["text"]
            }
            for seg in segments
        ]
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(data, jf, ensure_ascii=False, indent=2)

def format_hhmmss(seconds):
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows of the matrix to unit L2 norm."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def kmeans(matrix: np.ndarray, k: int = 2, n_iter: int = 20, seed: int = 0) -> np.ndarray:
    """Simple k-means clustering using NumPy."""
    rng = np.random.default_rng(seed)
    centroids = matrix[rng.choice(len(matrix), size=k, replace=False)]
    for _ in range(n_iter):
        distances = ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for i in range(k):
            if np.any(labels == i):
                centroids[i] = matrix[labels == i].mean(axis=0)
    return labels


def extract_phone_number(name: str) -> str | None:
    """Extract the first 11+ digit sequence from a filename."""
    import re
    match = re.search(r"(\d{11,})", name)
    return match.group(1) if match else None

# === 1. Чтение конфигурации ===
print("1. Чтение конфигурации...")
ENV_FILE = "/root/audio-lora-builder/config/env.vars"
WIN_AUDIO_SRC = None
HF_TOKEN = os.getenv("HF_TOKEN")

if os.path.exists(ENV_FILE):
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("WIN_AUDIO_SRC="):
                WIN_AUDIO_SRC = line.strip().split("=", 1)[1]
            elif line.startswith("HF_TOKEN=") and not HF_TOKEN:
                HF_TOKEN = line.strip().split("=", 1)[1]
if not WIN_AUDIO_SRC or not os.path.exists(WIN_AUDIO_SRC):
    print("⚠️ Не удалось найти корректный путь до папки с аудиофайлами.")
    user_input = input("Введите путь до папки с аудиофайлами [/mnt/c/]: ").strip()
    if not user_input:
        user_input = "/mnt/c/"
    WIN_AUDIO_SRC = user_input.replace("\\", "/")
    Path("/root/audio-lora-builder/config").mkdir(parents=True, exist_ok=True)
    with open(ENV_FILE, "w", encoding="utf-8") as f:
        f.write(f"WIN_AUDIO_SRC={WIN_AUDIO_SRC}\n")
        if HF_TOKEN:
            f.write(f"HF_TOKEN={HF_TOKEN}\n")
if not HF_TOKEN:
    print("⚠️ Не найден токен Hugging Face.")
    HF_TOKEN = input("Введите токен Hugging Face: ").strip()
    Path("/root/audio-lora-builder/config").mkdir(parents=True, exist_ok=True)
    with open(ENV_FILE, "a", encoding="utf-8") as f:
        f.write(f"HF_TOKEN={HF_TOKEN}\n")

# === 2. Конвертация файлов ===
print("2. 🎧 Конвертация аудиофайлов...")
AUDIO_EXTENSIONS = [".m4a", ".mp3", ".aac"]
SRC = Path(WIN_AUDIO_SRC)
DST = Path("/root/audio-lora-builder/input/audio_src")
if DST.exists(): shutil.rmtree(DST)
DST.mkdir(parents=True, exist_ok=True)
files = [f for f in SRC.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
for idx, file in enumerate(files, 1):
    relative = file.relative_to(SRC)
    output = DST / relative.with_suffix(".wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"🎛 ({idx}) {file} → {output}")
    subprocess.run(["ffmpeg", "-y", "-i", str(file), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(output)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"✅ Всего сконвертировано: {len(files)}")

# === 3. Обработка аудио ===
print("\n3. 🤖 Распознавание и диаризация...")
OUTPUT_DIR = Path("/root/audio-lora-builder/output")
if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

wav_files = list(DST.rglob("*.wav"))
if not wav_files:
    print("⚠️ Нет файлов для обработки")
    exit(0)

model = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
audio_reader = Audio(sample_rate=16000)
embedding_model = Model.from_pretrained("pyannote/embedding")
ks = None
for layer in embedding_model.sincnet.modules():
    if hasattr(layer, "kernel_size"):
        ks = layer.kernel_size
        break
if ks is None:
    raise RuntimeError("No layer with kernel_size found in SincNet")
min_len = ks if isinstance(ks, int) else ks[0]

all_embeddings = []
segment_map = {}

print("🔍 Обрабатываем файлы...")
start_all = time.time()

for idx, audio_path in enumerate(wav_files, 1):
    rel_path = audio_path.relative_to(DST)
    print(f"📝 ({idx}/{len(wav_files)}) {rel_path}")
    waveform, sample_rate = audio_reader(str(audio_path))

    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=2)
    segments, _ = model.transcribe(str(audio_path), language="ru", beam_size=5, vad_filter=True)
    segments = list(segments)

    emb_list = []
    seg_data = []

    for seg in diarization.itertracks(yield_label=True):
        turn, _, speaker = seg
        segment_audio = waveform[:, int(turn.start * sample_rate):int(turn.end * sample_rate)]
        segment_audio = (
            segment_audio if torch.is_tensor(segment_audio) else torch.from_numpy(segment_audio)
        )
        segment_tensor = segment_audio.unsqueeze(0).float()
        if segment_tensor.shape[-1] < min_len:
            segment_tensor = F.pad(segment_tensor, (0, min_len - segment_tensor.shape[-1]))
        if segment_tensor.shape[-1] < min_len:
            continue
        with torch.no_grad():
            emb_tensor = embedding_model(segment_tensor)
        emb_np = emb_tensor.squeeze(0).cpu().numpy()
        all_embeddings.append(emb_np)
        emb_list.append(emb_np)

        text = ""
        for s in segments:
            if s.start >= turn.start and s.end <= turn.end:
                text += s.text.strip() + " "
        seg_data.append({"start": turn.start, "end": turn.end, "text": text.strip(), "embedding": emb_np})

    segment_map[str(rel_path)] = seg_data

# Кластеризация всех эмбеддингов
emb_matrix = l2_normalize(np.vstack(all_embeddings).astype("float32"))
labels = kmeans(emb_matrix, k=2, n_iter=20)

# Определим какой кластер — ты
label_counts = Counter(labels)
you_cluster = label_counts.most_common(1)[0][0]

# Присваиваем роли и сохраняем
flat_index = 0
for rel_path, segments in segment_map.items():
    enriched = []
    caller_id = extract_phone_number(str(rel_path)) or "caller"
    for seg in segments:
        cluster = labels[flat_index]
        enriched.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "cluster": cluster
        })
        flat_index += 1
    write_json_v2(enriched, OUTPUT_DIR / Path(rel_path), rel_path, 0000000000000, caller_id)

total_time = format_hhmmss(time.time() - start_all)
print(f"✅ Обработка завершена. Всего файлов: {len(wav_files)}")
print(f"⏱️ Время выполнения: {total_time}")

# === Завершение ===
print("\n✅ Выполнение process_audio.py завершено.")
