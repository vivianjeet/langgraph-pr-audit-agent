"""
Benchmark single-text embed() (loop) vs embed_batch() on N throwaway texts.
Run: python -m scripts.bench_embed
Makes real API calls - uses the free tier, keep N modest
"""

import time
from src.db.vectorstore import embed, embed_batch, EMBED_DIM

N = 30
texts = [f"def handler_{i}(requests): return authorize(requests.user)" for i in range(N)]

def _time(label, fn):
    start = time.perf_counter()
    vectors = fn()
    elapsed = time.perf_counter() - start
    # Sanity: right count, right dimensionality
    assert len(vectors) == N, f"{label}: expected {N} vectors, got {len(vectors)}"
    assert all(len(v) == EMBED_DIM for v in vectors), f"{label}: wrong dim"
    print(f"{label:18} {elapsed:6.25f}s ({elapsed / N * 1000:5.1f} ms/text)")
    return elapsed

if __name__ == "__main__":
    print(f"Embedding (N) texts, EMBED_DIM={EMBED_DIM}\n")
    loop_t = _time("loop embed()", lambda: [embed(t) for t in texts])
    batch_t = _time("embed_batch()", lambda: embed_batch(texts))
    if batch_t > 0:
        print(f"\nbatch is {loop_t / batch_t:.1f}x faster ({N} texts, fewer round-trips)")
