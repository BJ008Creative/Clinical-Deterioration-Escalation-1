"""Section 9.2 (search_guidelines tool). Uses scikit-learn's TF-IDF instead
of a sentence-transformer + FAISS/ChromaDB index (per the tech stack table)
to keep the hackathon build light and dependency-free; swapping in real
embeddings later is a one-file change (see README "deviations" section for
why this is safe to defer). Guideline text lives in data/guidelines/*.md,
one chunk per paragraph, each with a citation id like the first line
"[NEWS2-3.2]"."""
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class GuidelineIndex:
    def __init__(self, guidelines_dir: str = "data/guidelines"):
        self.chunks = []      # list of {"id": str, "text": str}
        self._load(guidelines_dir)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        if self.chunks:
            self._matrix = self._vectorizer.fit_transform([c["text"] for c in self.chunks])
        else:
            self._matrix = None

    def _load(self, guidelines_dir: str):
        p = Path(guidelines_dir)
        if not p.exists():
            return
        for file in sorted(p.glob("*.md")):
            text = file.read_text()
            for block in re.split(r"\n\s*\n", text.strip()):
                block = block.strip()
                if not block:
                    continue
                m = re.match(r"\[([\w\-.]+)\]\s*(.*)", block, re.DOTALL)
                if m:
                    self.chunks.append({"id": m.group(1), "text": m.group(2).strip()})

    def search(self, query: str, top_k: int = 2) -> list:
        if self._matrix is None:
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(range(len(self.chunks)), key=lambda i: sims[i], reverse=True)
        return [self.chunks[i] for i in ranked[:top_k] if sims[i] > 0]
