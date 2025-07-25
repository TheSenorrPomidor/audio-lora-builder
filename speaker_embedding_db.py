import json
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.spatial.distance import cosine

# Optional imports are placed in a try block so that the module can be imported
# even if pyannote.audio is not installed. Functions depending on it will raise
# an ImportError when called.
try:
    from pyannote.audio import Audio, Model
    from pyannote.core import Segment
except Exception as e:  # pyannote might not be installed during linting
    Audio = None
    Model = None
    Segment = None


class SpeakerEmbeddingDB:
    """Simple persistent database for speaker embeddings."""

    def __init__(self, path: str = "speaker_db.json", threshold: float = 0.7, device: str = "cpu"):
        self.path = Path(path)
        self.threshold = threshold
        self.device = device
        self.embeddings: Dict[str, np.ndarray] = self._load()
        if Model is None or Audio is None:
            raise ImportError("pyannote.audio is required for SpeakerEmbeddingDB")
        self.audio = Audio(sample_rate=16000, mono=True)
        self.model = Model.from_pretrained("pyannote/embedding").to(device)

    # ------------------------------------------------------------------
    def _load(self) -> Dict[str, np.ndarray]:
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {spk: np.array(emb, dtype=np.float32) for spk, emb in raw.items()}
        return {}

    # ------------------------------------------------------------------
    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            serializable = {spk: emb.tolist() for spk, emb in self.embeddings.items()}
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    def _compute_embedding(self, audio_file: str, segment: Segment) -> np.ndarray:
        waveform, _ = self.audio.crop(audio_file, segment)
        emb = self.model(waveform.unsqueeze(0))
        emb = emb[0].detach().cpu().numpy()
        return emb / np.linalg.norm(emb)

    # ------------------------------------------------------------------
    def match(self, embedding: np.ndarray) -> str:
        best_id = None
        best_score = -1.0
        for spk, stored in self.embeddings.items():
            score = 1 - cosine(embedding, stored)
            if score > best_score:
                best_score = score
                best_id = spk
        if best_score >= self.threshold:
            return best_id
        return None

    # ------------------------------------------------------------------
    def add_embedding(self, speaker_id: str, embedding: np.ndarray) -> None:
        if speaker_id in self.embeddings:
            current = self.embeddings[speaker_id]
            self.embeddings[speaker_id] = (current + embedding) / 2
        else:
            self.embeddings[speaker_id] = embedding
        self._save()

    # ------------------------------------------------------------------
    def process_segment(self, audio_file: str, start: float, end: float) -> str:
        segment = Segment(start, end)
        embedding = self._compute_embedding(audio_file, segment)
        matched = self.match(embedding)
        if matched is not None:
            self.add_embedding(matched, embedding)
            return matched
        # assign new ID
        new_id = f"speaker_{len(self.embeddings) + 1}"
        self.add_embedding(new_id, embedding)
        return new_id
