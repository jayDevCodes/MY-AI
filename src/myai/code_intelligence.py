from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeSymbol:
    name: str
    kind: str
    path: str
    line: int
    end_line: int | None
    parent: str | None = None


@dataclass(frozen=True)
class CodeFile:
    path: str
    imports: tuple[str, ...]
    symbols: tuple[CodeSymbol, ...]


class CodeIntelligenceIndex:
    """Persistent AST/symbol graph with incremental freshness-aware invalidation."""

    snapshot_version = 4

    def __init__(self) -> None:
        self.files: dict[str, CodeFile] = {}
        self._snapshot_signatures: dict[str, tuple[int, int] | None] = {}
        self._snapshot_root: str | None = None

    def index_file(self, path: str | Path) -> CodeFile:
        file_path = Path(path)
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
        imports: list[str] = []
        symbols: list[CodeSymbol] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend(
                    f"{module}.{alias.name}" if module else alias.name
                    for alias in node.names
                )

        def visit(body: list[ast.stmt], parent: str | None = None) -> None:
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    symbols.append(
                        CodeSymbol(
                            name=node.name,
                            kind=kind,
                            path=str(file_path),
                            line=node.lineno,
                            end_line=node.end_lineno,
                            parent=parent,
                        )
                    )
                    visit(node.body, node.name)
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    visit(node.body, parent)

        visit(tree.body)
        code_file = CodeFile(
            path=str(file_path),
            imports=tuple(sorted(set(imports))),
            symbols=tuple(symbols),
        )
        self.files[str(file_path)] = code_file
        return code_file

    def _candidate_paths(self, root: str | Path) -> set[str]:
        root_path = Path(root)
        return {
            str(path)
            for path in root_path.rglob("*.py")
            if not any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.parts)
        }

    @staticmethod
    def _stat_signature(path: str | Path) -> tuple[int, int] | None:
        try:
            stat = Path(path).stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    @staticmethod
    def _module_candidates(module: str) -> set[str]:
        parts = module.split(".")
        return {"/".join(parts) + ".py", "/".join(parts) + "/__init__.py"}

    def _is_dependent_on(self, file_path: str, changed_path: str) -> bool:
        changed = Path(changed_path).resolve()
        source = Path(file_path).resolve()
        code_file = self.files.get(file_path)
        if code_file is None:
            return False
        root = Path(self._snapshot_root or ".").resolve()
        for module in code_file.imports:
            for candidate in self._module_candidates(module):
                if (root / candidate).resolve() == changed:
                    return True
        return source == changed

    def index_tree(self, root: str | Path) -> int:
        count = 0
        for path in self._candidate_paths(root):
            try:
                self.index_file(path)
                count += 1
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
        return count

    def refresh_if_stale(self, root: str | Path) -> bool:
        """Incrementally reindex changed files and their direct import dependents."""
        root_path = Path(root)
        current_paths = self._candidate_paths(root_path)
        current_signatures = {path: self._stat_signature(path) for path in current_paths}
        stale_paths = {
            path
            for path in current_paths
            if self._snapshot_signatures.get(path) != current_signatures[path]
        }
        removed_paths = set(self._snapshot_signatures) - current_paths
        stale_paths.update(removed_paths)

        if self._snapshot_root != str(root_path):
            self.files.clear()
            self.index_tree(root_path)
            self._snapshot_signatures = current_signatures
            self._snapshot_root = str(root_path)
            return True

        if not stale_paths:
            return False

        changed_existing = {path for path in stale_paths if path in current_paths}
        affected = set(changed_existing)
        for path in current_paths:
            if any(self._is_dependent_on(path, changed) for changed in changed_existing):
                affected.add(path)

        for path in removed_paths:
            self.files.pop(path, None)

        for path in affected:
            if path in current_paths:
                try:
                    self.index_file(path)
                except (OSError, SyntaxError, UnicodeDecodeError):
                    self.files.pop(path, None)

        self._snapshot_signatures = current_signatures
        self._snapshot_root = str(root_path)
        return True

    def search(self, query: str, limit: int = 20) -> tuple[CodeSymbol, ...]:
        terms = [term.casefold() for term in query.split() if term.strip()]
        if not terms:
            return ()
        matches: list[tuple[int, CodeSymbol]] = []
        for code_file in self.files.values():
            for symbol in code_file.symbols:
                haystack = f"{symbol.name} {symbol.kind} {symbol.path}".casefold()
                score = sum(term in haystack for term in terms)
                if score:
                    matches.append((score, symbol))
        matches.sort(key=lambda item: (-item[0], item[1].path, item[1].line))
        return tuple(symbol for _, symbol in matches[:limit])

    def context_map(self, query: str, limit: int = 8) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "path": symbol.path,
                "symbol": symbol.name,
                "kind": symbol.kind,
                "line": symbol.line,
                "end_line": symbol.end_line,
                "parent": symbol.parent,
            }
            for symbol in self.search(query, limit=limit)
        )

    def read_context(self, query: str, limit: int = 5, padding: int = 4) -> tuple[dict[str, object], ...]:
        """Read only the source ranges belonging to matched symbols."""
        contexts: list[dict[str, object]] = []
        for symbol in self.search(query, limit=limit):
            try:
                lines = Path(symbol.path).read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            start = max(1, symbol.line - padding)
            end = min(len(lines), (symbol.end_line or symbol.line) + padding)
            contexts.append(
                {
                    "path": symbol.path,
                    "symbol": symbol.name,
                    "start_line": start,
                    "end_line": end,
                    "text": "\n".join(lines[start - 1 : end]),
                }
            )
        return tuple(contexts)

    def save_snapshot(self, path: str | Path, root: str | Path = ".") -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        signatures = {
            file_path: self._stat_signature(file_path)
            for file_path in self.files
            if self._stat_signature(file_path) is not None
        }
        payload = {
            "version": self.snapshot_version,
            "root": str(Path(root)),
            "signatures": signatures,
            "files": [
                {
                    "path": code_file.path,
                    "imports": list(code_file.imports),
                    "symbols": [
                        {
                            "name": symbol.name,
                            "kind": symbol.kind,
                            "path": symbol.path,
                            "line": symbol.line,
                            "end_line": symbol.end_line,
                            "parent": symbol.parent,
                        }
                        for symbol in code_file.symbols
                    ],
                }
                for code_file in self.files.values()
            ],
        }
        target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        self._snapshot_signatures = signatures
        self._snapshot_root = str(Path(root))

    def load_snapshot(self, path: str | Path, root: str | Path = ".") -> bool:
        target = Path(path)
        if not target.exists():
            return False
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
            if payload.get("version") != self.snapshot_version:
                return False
            expected_root = str(Path(root))
            if str(payload.get("root", expected_root)) != expected_root:
                return False
            signatures = {
                str(key): tuple(value) if value is not None else None
                for key, value in payload.get("signatures", {}).items()
            }
            current_paths = self._candidate_paths(root)
            if current_paths != set(signatures):
                return False
            for file_path, signature in signatures.items():
                if self._stat_signature(file_path) != signature:
                    return False

            files: dict[str, CodeFile] = {}
            for item in payload.get("files", []):
                symbols = tuple(
                    CodeSymbol(
                        name=str(symbol["name"]),
                        kind=str(symbol["kind"]),
                        path=str(symbol["path"]),
                        line=int(symbol["line"]),
                        end_line=int(symbol["end_line"]) if symbol.get("end_line") is not None else None,
                        parent=str(symbol["parent"]) if symbol.get("parent") is not None else None,
                    )
                    for symbol in item.get("symbols", [])
                )
                code_file = CodeFile(
                    path=str(item["path"]),
                    imports=tuple(str(value) for value in item.get("imports", [])),
                    symbols=symbols,
                )
                files[code_file.path] = code_file
            self.files = files
            self._snapshot_signatures = signatures
            self._snapshot_root = expected_root
            return True
        except (OSError, ValueError, TypeError, KeyError):
            self.files.clear()
            return False
