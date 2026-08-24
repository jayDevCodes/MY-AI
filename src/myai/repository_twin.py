from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .code_intelligence import CodeIntelligenceIndex


@dataclass(frozen=True)
class TwinNode:
    key: str
    kind: str
    path: str
    symbol: str | None = None
    line: int | None = None
    end_line: int | None = None
    stability: float = 0.5
    verification_confidence: float = 0.0


@dataclass(frozen=True)
class TwinEdge:
    source: str
    target: str
    kind: str


@dataclass(frozen=True)
class ImpactSlice:
    center: str
    nodes: tuple[TwinNode, ...]
    edges: tuple[TwinEdge, ...]
    source_context: tuple[dict[str, object], ...]


class CausalRepositoryTwin:
    """Evidence-oriented repository graph for targeted diagnosis and repair."""

    def __init__(self, code_index: CodeIntelligenceIndex) -> None:
        self.code_index = code_index
        self.nodes: dict[str, TwinNode] = {}
        self.edges: set[TwinEdge] = set()
        self._importers: dict[str, set[str]] = {}
        self._indexed_root: str | None = None

    def rebuild(self, root: str | Path) -> None:
        root_path = Path(root)
        self.nodes.clear()
        self.edges.clear()
        self._importers.clear()
        self._indexed_root = str(root_path)

        for code_file in self.code_index.files.values():
            file_key = self._file_key(code_file.path)
            self.nodes[file_key] = TwinNode(
                key=file_key,
                kind="file",
                path=code_file.path,
                stability=0.8,
            )
            for symbol in code_file.symbols:
                symbol_key = self._symbol_key(symbol.path, symbol.name, symbol.line)
                self.nodes[symbol_key] = TwinNode(
                    key=symbol_key,
                    kind=symbol.kind,
                    path=symbol.path,
                    symbol=symbol.name,
                    line=symbol.line,
                    end_line=symbol.end_line,
                    stability=0.8,
                )
                self.edges.add(TwinEdge(symbol_key, file_key, "declared_in"))

            for imported in code_file.imports:
                target = self._resolve_import(code_file.path, imported, root_path)
                if target is None:
                    continue
                self.edges.add(TwinEdge(file_key, target, "imports"))
                self._importers.setdefault(target, set()).add(file_key)

    def affected_files(self, path: str | Path) -> tuple[str, ...]:
        center = self._file_key(path)
        seen: set[str] = set()
        queue = [center]
        while queue:
            current = queue.pop()
            if current in seen:
                continue
            seen.add(current)
            queue.extend(self._importers.get(current, ()))
        return tuple(sorted(self.nodes[key].path for key in seen if key in self.nodes))

    def impact_slice(self, query: str, *, limit: int = 8, padding: int = 4) -> ImpactSlice:
        symbols = self.code_index.search(query, limit=limit)
        if not symbols:
            return ImpactSlice(query, (), (), ())

        center = self._symbol_key(symbols[0].path, symbols[0].name, symbols[0].line)
        keys = {self._file_key(symbol.path) for symbol in symbols}
        keys.update(self._symbol_key(symbol.path, symbol.name, symbol.line) for symbol in symbols)
        for symbol in symbols:
            file_key = self._file_key(symbol.path)
            keys.update(
                edge.source
                for edge in self.edges
                if edge.target == file_key and edge.kind == "imports"
            )

        nodes = tuple(self.nodes[key] for key in sorted(keys) if key in self.nodes)
        edges = tuple(edge for edge in self.edges if edge.source in keys or edge.target in keys)
        source_context = self.code_index.read_context(query, limit=limit, padding=padding)
        return ImpactSlice(center, nodes, edges, source_context)

    @staticmethod
    def _file_key(path: str | Path) -> str:
        return f"file:{Path(path)}"

    @staticmethod
    def _symbol_key(path: str | Path, name: str, line: int) -> str:
        return f"symbol:{Path(path)}:{name}:{line}"

    @staticmethod
    def _resolve_import(path: str, imported: str, root: Path) -> str | None:
        parts = imported.split(".")
        current = Path(path).parent
        candidates = [current.joinpath(*parts).with_suffix(".py"), root.joinpath(*parts).with_suffix(".py")]
        candidates += [current.joinpath(*parts, "__init__.py"), root.joinpath(*parts, "__init__.py")]
        for candidate in candidates:
            if candidate.exists():
                return CausalRepositoryTwin._file_key(candidate)
        return None
