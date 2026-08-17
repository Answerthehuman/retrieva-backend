# Runtime profiles

Retrieva trades RAM for retrieval quality through a profile chosen at launch.
Pick the tier your hardware can carry; move up as it improves.

```bash
make up-lite       # ~2 GB  — default, and the design target on a 5.9 GB host
make up-standard   # ~4 GB  — + reranking + multi-query
make up-full       # 8 GB+  — + contextual retrieval
```

`/health` reports the active profile and the features it enabled, so the UI can
explain *why* results are less refined rather than leaving it looking like a fault.

---

## The one thing that is NOT a profile setting

**The embedding model is fixed across all profiles, on purpose.**

Milvus bakes the vector width into the collection when it is created:

```python
FieldSchema(name="embedding", dtype=FLOAT_VECTOR, dim=dim)  # fixed forever
```

If the embedding model varied per profile, switching from `lite` (768-dim
`nomic-embed-text`) to `full` (1024-dim `bge-m3`) would leave every stored
vector the wrong width. Inserts would fail with an opaque pymilvus error, and —
worse — searches against a mismatched index return *plausible nonsense* rather
than an error.

So: **choose an embedding model that runs comfortably in `lite`, and keep it.**
The levers that scale with your RAM are reranking, multi-query expansion, and
contextual retrieval — all of which are either stateless or affect only
newly-ingested documents.

A startup check compares the configured dimension against the live collection
and reports a mismatch loudly (in logs and in `/health`) with the exact command
to fix it. It deliberately does **not** crash the process: a restart loop would
make `/health` unreachable precisely when you need it to diagnose the problem.

**Changing the embedding model later means re-ingesting everything.** That's a
migration, not a config change — do it deliberately.

---

## What each profile enables

| | **lite** | **standard** | **full** |
|---|---|---|---|
| Target RAM | ~2 GB | ~4 GB | 8 GB+ |
| Embedding model | *identical across all three* | ← | ← |
| Cross-encoder reranking | off | on | on |
| Multi-query expansion | off | on | on |
| Contextual retrieval | off | off | on |
| `retrieval_top_k` | 30 | 50 | 100 |
| `rerank_top_k` | 10 | 10 | 20 |
| Agent max tool calls | 4 | 6 | 10 |
| Ingest batch size | 16 | 32 | 64 |
| Milvus memory cap | 512 MB | 1 GB | 2 GB |
| Langfuse | Cloud | Cloud | self-host possible |

### Why these particular features are the profile knobs

- **Reranking** needs `sentence-transformers` + torch (~200 MB resident). It is
  the best quality-per-MB lever available, which is why it is the first thing
  to switch on once there is headroom.
- **Multi-query expansion** costs an extra LLM round trip plus N extra searches
  per question — latency and tokens rather than RAM.
- **Contextual retrieval** makes one LLM call *per chunk* at ingest time. It is
  the largest quality gain available and by far the most expensive; `full` only.
- **Ingest batch size** is the direct peak-memory lever during ingestion.

---

## Precedence: explicit settings always win

A profile supplies **defaults**, not overrides. Anything you set explicitly —
in `.env`, as an env var, or in code — beats the profile:

```
explicit env var  >  .env entry  >  profile default  >  field default
```

So this enables reranking even in `lite`:

```bash
RERANK_ENABLED=true make up-lite
```

⚠️ The corollary bites: **an entry left uncommented in `.env` silently pins that
value and the profile can never change it.** The profile-derived keys ship
commented out in `.env.example` for exactly this reason:

```bash
# ── Profile-derived (leave commented to let RETRIEVA_PROFILE govern) ──
# RERANK_ENABLED=false
# RETRIEVAL_TOP_K=50
```

If a profile switch seems to do nothing, check whether the key is pinned in
`.env` first — that is the usual cause.

---

## Enabling reranking (standard / full)

Reranking needs a package that is not in the default image. Build it in once:

```bash
make build-rerank        # docker compose build --build-arg INSTALL_RERANK=true backend
make up-standard
```

Without that build argument, `RERANK_ENABLED=true` logs an explicit warning and
leaves results unranked rather than failing — but you get no reranking, so the
warning matters.

---

## Evaluating across profiles

Eval results are only comparable **within** a profile. `lite` has reranking off,
so its numbers will be lower by construction — that is not a regression.

`scripts/eval_retrieval.py` records the profile and embedding model in every
results file and warns loudly when comparing across a mismatch:

```bash
make eval-baseline   # record a baseline for the CURRENT profile
make eval            # compare against it
```

Keep one baseline per profile, e.g. `evals/results/baseline-lite.json`.

---

## After a hardware upgrade

1. Raise the WSL2 memory cap in `~/.wslconfig`, then `wsl --shutdown`.
2. `make build-rerank` (once).
3. `make up-full`.
4. Record a fresh baseline: `make eval-baseline`.

The only thing that needs re-ingesting is contextual retrieval, and only if you
want it applied to documents already indexed — it takes effect automatically for
anything ingested afterwards.
