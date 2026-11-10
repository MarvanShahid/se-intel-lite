# Architecture

```mermaid
flowchart LR
  subgraph Ingest
    R[Python repo @ pinned commit] --> PARSE[AST parser<br/>graph/parser.py]
    R --> GIT[git history miner<br/>history/mining.py]
  end
  PARSE --> G[(Code graph<br/>dir/file/class/function<br/>contains/imports/invokes/inherits)]
  G --> EI[Entity index<br/>exact id / name / BM25 ids / BM25 code]
  G --> CI[Chunk index<br/>BM25 + TF-IDF RRF]
  G --> GF[File dependency graph<br/>fan-in, fan-out, PageRank]
  GIT --> HF[Change features + bug-fix labels<br/>time-based train/tune/test]
  GF --> HS[Hotspot ranker  G2<br/>quality/hotspot.py]
  HF --> RM[Risk model  G3<br/>logistic regression + Platt<br/>risk/model.py]
  GF --> RM

  subgraph Agent [ReAct agent  agent/react.py]
    LLM[LLM: Claude] <--> LOOP[Thought → Action → Observation<br/>step budget 12, state: visited / retrieved / tool log]
    LOOP --> VER[Citation verifier  G1]
  end
  EI --> T1[search_entity]
  G --> T2[traverse_graph  G1]
  G --> T3[retrieve_entity]
  HS --> T4[run_quality / get_hotspots  G2]
  RM --> T5[get_risk  G3]
  T1 & T2 & T3 & T4 & T5 <--> LOOP
  VER --> OUT[Answer + evidence trail<br/>+ trajectory log]

  subgraph Baselines
    B0[B0 plain LLM]
    B1[B1 flat RAG over chunk index]
    B2[B2 LocAgent-style: 3 graph tools, plain protocol]
  end
  CI --> B1
```

| Component | File | Gap | Switch (ablation) |
|---|---|---|---|
| Typed multi-hop traversal | `tools/graph_tools.py` | 1 | `A1_no_traverse`, `A2_hop1` |
| QA protocol prompt | `agent/react.py` (`QA_PROTOCOL`) | 1 | `A7_no_qa_prompt` |
| Citation verifier (one revision) | `agent/react.py` (`_verify`) | 1 | `A8_no_verifier` |
| RunQuality / GetHotspots + graph-aware hotspot score | `tools/quality_tools.py`, `quality/hotspot.py` | 2 | `A3_no_quality`; ranking without graph = `cc` |
| GetRisk, calibrated graph-aware risk model | `tools/quality_tools.py`, `risk/model.py` | 3 | `A3b_no_risk`; model variants `metrics` (no graph), `full_uncalibrated` |

All system configurations are declared in `agent/factory.py` (`SYSTEMS`).

**Robustness.** Tools are read-only and never raise into the loop (errors become observations). Invalid JSON
replies get a corrective observation. LLM/API errors are retried by the SDK (5 retries) and then recorded in the
trace; three consecutive errors end the run with an explicit message. When the step budget is spent the agent
is told to finish, and `budget_exhausted` is logged. Cited ids that never appeared in an observation are dropped
from the evidence and counted (`ungrounded`). All LLM calls are cached on disk, so a crashed experiment resumes
where it stopped, and re-running it reproduces the saved trajectories exactly.
