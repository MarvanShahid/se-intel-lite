"""Write results/graph_stats.json (nodes/edges per type for each frozen repository)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seintel.eval.dataset import ROOT, load_config
from seintel.graph.store import RepoGraph

cfg = load_config()
out = {}
for r in cfg["repos"]:
    rg = RepoGraph.from_repo(ROOT / cfg["repos_dir"] / r["name"])
    s = rg.stats()
    b = s["builder"]
    s["heuristic_invoke_share"] = round(b.get("heuristic_invoke_edges", 0) / max(1, b.get("invoke_edges", 0)), 3)
    out[r["name"]] = s
    print(r["name"], s["nodes"], s["edges"], "heuristic share", s["heuristic_invoke_share"])
(ROOT / "results" / "graph_stats.json").write_text(json.dumps(out, indent=2))
