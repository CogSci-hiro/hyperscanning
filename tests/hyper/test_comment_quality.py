"""Maintainability guard tests for source comments and documentation.

These tests enforce a baseline level of explanatory context in each source
module so future contributors can understand purpose and intent quickly.
"""

import ast
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "hyper"


def _python_files() -> list[Path]:
    """Collect all project source files under src/hyper."""
    return sorted(SRC_ROOT.rglob("*.py"))


def test_every_source_module_has_context_comments_or_module_docstring() -> None:
    """Each module should carry purpose-level documentation at file scope."""
    offenders = []

    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        comment_lines = [line for line in text.splitlines() if line.strip().startswith("#")]
        module = ast.parse(text)
        module_doc = ast.get_docstring(module)

        # Accept either: explicit module docstring, or multiple top comments
        # that convey purpose/structure in header form.
        if not module_doc and len(comment_lines) < 3:
            offenders.append(str(path))

    assert not offenders, "Missing high-level context comments/docstrings:\n" + "\n".join(offenders)


def test_non_trivial_modules_include_docstrings_on_public_members() -> None:
    """Public functions/classes should carry docstrings for readability."""
    offenders = []

    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)

        public_defs = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_")
        ]

        # Tiny modules may only export constants and need no function/class docs.
        if not public_defs:
            continue

        for node in public_defs:
            if ast.get_docstring(node) is None:
                offenders.append(f"{path}:{node.name}")

    assert not offenders, "Missing public API docstrings:\n" + "\n".join(offenders)
