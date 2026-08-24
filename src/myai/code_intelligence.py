from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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
    """Lightweight AST index for narrow, repeatable code-context retrieval."""

    def __init__(self) -> None:
        self.files: dict[str, CodeFile] = {}

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

        def visit(body: Iterable[ast.AST], parent: str | None = None) -> None:
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    symbols.append(
                        CodeSymbol(
                            name=node.name,
                            kind=kind,
                            path=str(file_path),
                            line=node.lineno,
                            end_line=getattr(node, "end_lineno", None),
                            parent=parent,
                        )
                    )
                    visit(getattr(node, "body", ()), node.name)
                elif isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    visit(getattr(node, "body", ()), parent)

        visit(tree.body)
        code_file = CodeFile(
            path=str(file_path),
            imports=tuple(sorted(set(imports))),
            symbols=tuple(symbols),
        )
        self.files[str(file_path)] = code_file
        return code_file

    def index_tree(self, root: str | Path) -> int:
        root_path = Path(root)
        count = 0
        for path in root_path.rglob("*.py"):
            if any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.parts):
                continue
            try:
                self.index_file(path)
                count += 1
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
        return count

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
