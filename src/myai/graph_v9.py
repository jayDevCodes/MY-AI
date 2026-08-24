from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .code_intelligence import CodeIntelligenceIndex


@dataclass(frozen=True)
class ProgramNode:
    key: str
    kind: str
    path: str
    symbol: str | None = None
    line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class ProgramEdge:
    source: str
    target: str
    kind: str


@dataclass(frozen=True)
class ProgramSlice:
    center: str
    nodes: tuple[ProgramNode, ...]
    edges: tuple[ProgramEdge, ...]
    source_context: tuple[dict[str, object], ...]


class ProgramGraph:
    """Unified structural, call, data-flow and control-flow graph for Python repositories."""

    def __init__(self, code_index: CodeIntelligenceIndex) -> None:
        self.code_index = code_index
        self.nodes: dict[str, ProgramNode] = {}
        self.edges: set[ProgramEdge] = set()
        self._symbols: dict[tuple[str, str], str] = {}

    def build(self, root: str | Path) -> None:
        del root
        self.nodes.clear()
        self.edges.clear()
        self._symbols.clear()

        for code_file in self.code_index.files.values():
            file_key = f"file:{Path(code_file.path)}"
            self.nodes[file_key] = ProgramNode(file_key, "file", code_file.path)
            for symbol in code_file.symbols:
                key = f"symbol:{Path(symbol.path)}:{symbol.name}:{symbol.line}"
                self.nodes[key] = ProgramNode(
                    key,
                    symbol.kind,
                    symbol.path,
                    symbol.name,
                    symbol.line,
                    symbol.end_line,
                )
                self._symbols[(symbol.path, symbol.name)] = key
                self.edges.add(ProgramEdge(key, file_key, "declared_in"))

        for code_file in self.code_index.files.values():
            for symbol in code_file.symbols:
                key = self._symbols.get((symbol.path, symbol.name))
                if not key:
                    continue
                try:
                    tree = ast.parse(Path(symbol.path).read_text(encoding="utf-8"), filename=symbol.path)
                except (OSError, SyntaxError, UnicodeDecodeError):
                    continue
                target = self._find_symbol_node(tree, symbol.name, symbol.line)
                if target is None:
                    continue
                for node in ast.walk(target):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        candidate = self._symbols.get((symbol.path, node.func.id))
                        if candidate and candidate != key:
                            self.edges.add(ProgramEdge(key, candidate, "calls"))
                    elif isinstance(node, ast.Name):
                        candidate = self._symbols.get((symbol.path, node.id))
                        if candidate and candidate != key:
                            self.edges.add(ProgramEdge(key, candidate, "data_flow"))
                    elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                        control_key = (
                            f"control:{symbol.path}:{symbol.line}:"
                            f"{type(node).__name__}:{getattr(node, 'lineno', 0)}"
                        )
                        self.nodes.setdefault(
                            control_key,
                            ProgramNode(control_key, "control", symbol.path, symbol.name, symbol.line, symbol.end_line),
                        )
                        self.edges.add(ProgramEdge(key, control_key, "control_flow"))

    @staticmethod
    def _find_symbol_node(tree: ast.AST, name: str, line: int) -> ast.AST | None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == name and getattr(node, "lineno", None) == line:
                    return node
        return None

    def slice(self, query: str, limit: int = 8) -> ProgramSlice:
        matches = self.code_index.search(query, limit=limit)
        if not matches:
            return ProgramSlice(query, (), (), ())
        center = f"symbol:{Path(matches[0].path)}:{matches[0].name}:{matches[0].line}"
        keys = {center}
        for symbol in matches:
            key = f"symbol:{Path(symbol.path)}:{symbol.name}:{symbol.line}"
            keys.add(key)
            keys.update(edge.target for edge in self.edges if edge.source == key)
            keys.update(edge.source for edge in self.edges if edge.target == key)
        nodes = tuple(self.nodes[key] for key in keys if key in self.nodes)
        edges = tuple(edge for edge in self.edges if edge.source in keys or edge.target in keys)
        context = self.code_index.read_context(query, limit=limit)
        return ProgramSlice(center, nodes, edges, context)
