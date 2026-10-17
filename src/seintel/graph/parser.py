"""AST based code graph builder (follows the LocAgent schema [1]).

Nodes : directory, file, class, function (methods are functions inside a class)
Edges : contains, imports, invokes, inherits
Node ids: directory -> "pkg/sub", file -> "pkg/sub/mod.py",
          class/function -> "pkg/sub/mod.py:Class.method"

Static analysis only: dynamic dispatch, decorators and runtime imports can be
missed. Edges that rely on a "globally unique method name" guess are flagged
with heuristic=True so the tools and the evaluation can filter them.
"""
from __future__ import annotations

import ast
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

SKIP_DIRS = {".git", ".tox", ".venv", "venv", "node_modules", "build", "dist",
             "__pycache__", ".eggs", ".mypy_cache", ".pytest_cache"}


@dataclass
class FileInfo:
    path: str
    source: str
    tree: ast.AST | None
    # alias -> ("module", file_path) | ("name", file_path, symbol)
    imports: dict = field(default_factory=dict)


def _module_candidates(rel_path: str) -> list[str]:
    """All dotted names under which a file could be imported (suffixes of its path)."""
    p = rel_path[:-3] if rel_path.endswith(".py") else rel_path
    parts = p.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return [".".join(parts[i:]) for i in range(len(parts)) if parts[i:]]


class CodeGraphBuilder:
    def __init__(self, repo_root: str | Path):
        self.root = Path(repo_root).resolve()
        self.g = nx.MultiDiGraph()
        self.files: dict[str, FileInfo] = {}
        self.mod_index: dict[str, str] = {}
        self.stats: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ build
    def build(self) -> nx.MultiDiGraph:
        self._collect_files()
        self._build_module_index()
        for fi in self.files.values():
            self._add_file_nodes(fi)
        for fi in self.files.values():
            self._collect_imports(fi)
        self._add_import_edges()
        self._add_inherit_edges()
        self._add_invoke_edges()
        self.g.graph["root"] = str(self.root)
        self.g.graph["stats"] = dict(self.stats)
        return self.g

    # ------------------------------------------------------------ collection
    def _collect_files(self) -> None:
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in SKIP_DIRS and not d.endswith(".egg-info"))
            for fn in sorted(filenames):
                if not fn.endswith(".py"):
                    continue
                full = Path(dirpath) / fn
                rel = full.relative_to(self.root).as_posix()
                src = full.read_text(encoding="utf-8", errors="replace")
                try:
                    tree = ast.parse(src)
                except (SyntaxError, ValueError, RecursionError):
                    self.stats["parse_failures"] += 1
                    tree = None
                self.files[rel] = FileInfo(rel, src, tree)

    def _build_module_index(self) -> None:
        cands: dict[str, list[str]] = defaultdict(list)
        for rel in self.files:
            for c in _module_candidates(rel):
                cands[c].append(rel)
        # ambiguous dotted names resolve to the shortest path (closest to repo root)
        self.mod_index = {name: min(paths, key=len) for name, paths in cands.items()}

    # ---------------------------------------------------------------- nodes
    def _add_dir_chain(self, rel_file: str) -> str:
        parts = rel_file.split("/")[:-1]
        parent = "."
        if "." not in self.g:
            self.g.add_node(".", type="directory", name=".", path=".")
        for i in range(len(parts)):
            d = "/".join(parts[: i + 1])
            if d not in self.g:
                self.g.add_node(d, type="directory", name=parts[i], path=d)
                self.g.add_edge(parent, d, type="contains")
            parent = d
        return parent

    def _add_file_nodes(self, fi: FileInfo) -> None:
        parent = self._add_dir_chain(fi.path)
        self.g.add_node(fi.path, type="file", name=os.path.basename(fi.path), path=fi.path,
                        start_line=1, end_line=fi.source.count("\n") + 1)
        self.g.add_edge(parent, fi.path, type="contains")
        self.stats["files"] += 1
        if fi.tree is not None:
            self._visit_body(fi, fi.tree.body, fi.path, qual="", in_class=False)

    def _visit_body(self, fi: FileInfo, body, parent_id: str, qual: str, in_class: bool) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                q = f"{qual}.{node.name}" if qual else node.name
                nid = f"{fi.path}:{q}"
                self.g.add_node(nid, type="class", name=node.name, qualname=q, path=fi.path,
                                start_line=node.lineno, end_line=getattr(node, "end_lineno", node.lineno),
                                bases=[ast.unparse(b) for b in node.bases])
                self.g.add_edge(parent_id, nid, type="contains")
                self.stats["classes"] += 1
                self._visit_body(fi, node.body, nid, q, in_class=True)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                q = f"{qual}.{node.name}" if qual else node.name
                nid = f"{fi.path}:{q}"
                self.g.add_node(nid, type="function", name=node.name, qualname=q, path=fi.path,
                                start_line=node.lineno, end_line=getattr(node, "end_lineno", node.lineno),
                                is_method=in_class)
                self.g.add_edge(parent_id, nid, type="contains")
                self.stats["functions"] += 1
                self._visit_body(fi, node.body, nid, q, in_class=False)
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                # definitions inside guarded blocks (try/except ImportError, if TYPE_CHECKING, ...)
                for sub in ("body", "orelse", "finalbody"):
                    self._visit_body(fi, getattr(node, sub, []) or [], parent_id, qual, in_class)
                for h in getattr(node, "handlers", []) or []:
                    self._visit_body(fi, h.body, parent_id, qual, in_class)

    # -------------------------------------------------------------- imports
    def _resolve_module(self, dotted: str, from_file: str, level: int = 0) -> str | None:
        if level:
            pkg = from_file.split("/")[:-1]
            if level > 1:
                pkg = pkg[: max(0, len(pkg) - (level - 1))]
            parts = pkg + ([p for p in dotted.split(".") if p] if dotted else [])
            p = "/".join(parts)
            for guess in (f"{p}.py", f"{p}/__init__.py"):
                if guess in self.files:
                    return guess
            return None
        return self.mod_index.get(dotted)

    def _collect_imports(self, fi: FileInfo) -> None:
        if fi.tree is None:
            return
        for node in ast.walk(fi.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    tgt = self._resolve_module(a.name, fi.path)
                    if not tgt:
                        continue
                    if a.asname:
                        fi.imports[a.asname] = ("module", tgt)
                    elif "." not in a.name:
                        fi.imports[a.name] = ("module", tgt)
                    else:  # "import a.b.c" binds the top-level package name "a"
                        top = self._resolve_module(a.name.split(".")[0], fi.path)
                        if top:
                            fi.imports[a.name.split(".")[0]] = ("module", top)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                mod_file = self._resolve_module(mod, fi.path, node.level)
                for a in node.names:
                    alias = a.asname or a.name
                    sub_dotted = f"{mod}.{a.name}" if mod else a.name
                    sub_file = self._resolve_module(sub_dotted, fi.path, node.level)
                    if sub_file and sub_file != mod_file:      # "from pkg import submodule"
                        fi.imports[alias] = ("module", sub_file)
                    elif mod_file:                            # "from mod import Name"
                        fi.imports[alias] = ("name", mod_file, a.name)

    def _add_import_edges(self) -> None:
        for fi in self.files.values():
            seen = set()
            for spec in fi.imports.values():
                tgt = spec[1]
                if tgt != fi.path and tgt not in seen:
                    self.g.add_edge(fi.path, tgt, type="imports")
                    self.stats["import_edges"] += 1
                    seen.add(tgt)

    # ------------------------------------------------------------- symbols
    def _lookup_in_file(self, path: str, name: str) -> str | None:
        """Top-level class/function `name` in `path` (follows one re-export hop)."""
        nid = f"{path}:{name}"
        if nid in self.g:
            return nid
        fi = self.files.get(path)
        if fi and name in fi.imports and fi.imports[name][0] == "name":
            nid2 = f"{fi.imports[name][1]}:{fi.imports[name][2]}"
            if nid2 in self.g:
                return nid2
        return None

    def _resolve_name(self, fi: FileInfo, name: str) -> str | None:
        nid = self._lookup_in_file(fi.path, name)
        if nid:
            return nid
        spec = fi.imports.get(name)
        if spec and spec[0] == "name":
            return self._lookup_in_file(spec[1], spec[2])
        return None

    def _add_inherit_edges(self) -> None:
        for nid, data in list(self.g.nodes(data=True)):
            if data.get("type") != "class":
                continue
            fi = self.files[data["path"]]
            for b in data.get("bases", []):
                tgt = None
                if "." not in b:
                    tgt = self._resolve_name(fi, b)
                else:
                    head, _, rest = b.partition(".")
                    spec = fi.imports.get(head)
                    if spec and spec[0] == "module" and "." not in rest:
                        tgt = self._lookup_in_file(spec[1], rest)
                if tgt and tgt != nid:
                    self.g.add_edge(nid, tgt, type="inherits")
                    self.stats["inherit_edges"] += 1

    # -------------------------------------------------------------- invokes
    def _methods_of(self, class_id: str, seen=None) -> dict[str, str]:
        """method name -> node id, including inherited methods."""
        seen = seen if seen is not None else set()
        if class_id in seen:
            return {}
        seen.add(class_id)
        out: dict[str, str] = {}
        for _, child, d in self.g.out_edges(class_id, data=True):
            if d["type"] == "contains" and self.g.nodes[child]["type"] == "function":
                out[self.g.nodes[child]["name"]] = child
        for _, parent, d in self.g.out_edges(class_id, data=True):
            if d["type"] == "inherits":
                for k, v in self._methods_of(parent, seen).items():
                    out.setdefault(k, v)
        return out

    def _add_invoke_edges(self) -> None:
        by_name: dict[str, list[str]] = defaultdict(list)
        for nid, d in self.g.nodes(data=True):
            if d.get("type") == "function":
                by_name[d["name"]].append(nid)
        # index function AST nodes once per file
        ast_index: dict[tuple[str, int, str], ast.AST] = {}
        for fi in self.files.values():
            if fi.tree is None:
                continue
            for n in ast.walk(fi.tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ast_index[(fi.path, n.lineno, n.name)] = n
        for nid, d in list(self.g.nodes(data=True)):
            if d.get("type") != "function":
                continue
            fi = self.files[d["path"]]
            node = ast_index.get((d["path"], d["start_line"], d["name"]))
            if node is None:
                continue
            cls_id = nid.rsplit(".", 1)[0] if d.get("is_method") else None
            targets: dict[str, bool] = {}   # target -> exact?
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                tgt, heur = self._resolve_call(fi, nid, cls_id, call.func, by_name)
                if tgt and tgt != nid:
                    targets[tgt] = targets.get(tgt, True) and not heur
                elif tgt is None:
                    self.stats["unresolved_calls"] += 1
            for t, exact in targets.items():
                self.g.add_edge(nid, t, type="invokes", heuristic=not exact)
                self.stats["invoke_edges"] += 1
                if not exact:
                    self.stats["heuristic_invoke_edges"] += 1

    def _resolve_call(self, fi, caller_id, cls_id, func, by_name):
        """Return (target_id | None, is_heuristic)."""
        if isinstance(func, ast.Name):
            nm = func.id
            nested = f"{fi.path}:{self.g.nodes[caller_id]['qualname']}.{nm}"
            if nested in self.g:
                return nested, False
            return self._resolve_name(fi, nm), False
        if isinstance(func, ast.Attribute):
            attr, base = func.attr, func.value
            if isinstance(base, ast.Name) and base.id in ("self", "cls") and cls_id in self.g:
                m = self._methods_of(cls_id).get(attr)
                if m:
                    return m, False
            if isinstance(base, ast.Name):
                spec = fi.imports.get(base.id)
                if spec and spec[0] == "module":
                    t = self._lookup_in_file(spec[1], attr)
                    if t:
                        return t, False
                cls = self._resolve_name(fi, base.id)
                if cls and self.g.nodes[cls]["type"] == "class":
                    m = self._methods_of(cls).get(attr)
                    if m:
                        return m, False
            # obj.method(): only when the method name is globally unique (flagged heuristic)
            cands = [c for c in by_name.get(attr, []) if self.g.nodes[c].get("is_method")]
            if len(cands) == 1 and len(attr) > 3 and not attr.startswith("__"):
                return cands[0], True
        return None, False


def build_graph(repo_root: str | Path) -> nx.MultiDiGraph:
    return CodeGraphBuilder(repo_root).build()
