"""
DAIMON Code Loader - Source code files with syntax awareness.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseLoader, Document

# Language detection by extension
LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".cs": "csharp",
    ".sql": "sql",
    ".vim": "vim",
    ".el": "elisp",
}

# Comment patterns by language
COMMENT_PATTERNS: Dict[str, Dict[str, str]] = {
    "python": {"line": r"#.*$", "block": r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''},
    "javascript": {"line": r"//.*$", "block": r"/\*[\s\S]*?\*/"},
    "typescript": {"line": r"//.*$", "block": r"/\*[\s\S]*?\*/"},
    "go": {"line": r"//.*$", "block": r"/\*[\s\S]*?\*/"},
    "rust": {"line": r"//.*$", "block": r"/\*[\s\S]*?\*/"},
    "java": {"line": r"//.*$", "block": r"/\*[\s\S]*?\*/"},
    "c": {"line": r"//.*$", "block": r"/\*[\s\S]*?\*/"},
    "cpp": {"line": r"//.*$", "block": r"/\*[\s\S]*?\*/"},
    "ruby": {"line": r"#.*$", "block": r"=begin[\s\S]*?=end"},
    "php": {"line": r"//.*$|#.*$", "block": r"/\*[\s\S]*?\*/"},
    "bash": {"line": r"#.*$", "block": ""},
    "zsh": {"line": r"#.*$", "block": ""},
    "lua": {"line": r"--.*$", "block": r"--\[\[[\s\S]*?\]\]"},
    "sql": {"line": r"--.*$", "block": r"/\*[\s\S]*?\*/"},
}


class CodeLoader(BaseLoader):
    """
    Loader for source code files.

    Features:
    - Language detection by extension
    - Optional comment extraction
    - Optional code stripping (keep only comments/docs)
    - Function/class detection
    """

    def __init__(
        self,
        extract_comments: bool = False,
        strip_comments: bool = False,
    ):
        """
        Initialize code loader.

        Args:
            extract_comments: If True, include extracted comments in metadata.
            strip_comments: If True, return only comments (for documentation).
        """
        self.extract_comments = extract_comments
        self.strip_comments = strip_comments

    def load(self, path: str) -> Document:
        """
        Load source code file.

        Args:
            path: Path to source file.

        Returns:
            Document with code content.
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = file_path.read_text(encoding="utf-8", errors="replace")

        # Detect language
        ext = file_path.suffix.lower()
        language = LANGUAGE_MAP.get(ext, "text")

        # Extract metadata
        metadata = {
            "format": "code",
            "language": language,
            "extension": ext,
            "size_bytes": file_path.stat().st_size,
            "line_count": text.count("\n") + 1,
        }

        # Extract comments if requested
        if self.extract_comments or self.strip_comments:
            comments = self._extract_comments(text, language)
            metadata["comments"] = comments

            if self.strip_comments:
                text = "\n".join(comments)

        # Extract structure info
        structure = self._extract_structure(text, language)
        if structure:
            metadata["structure"] = structure

        text = self._clean_text(text)

        return Document(
            text=text,
            source=str(file_path.absolute()),
            title=file_path.name,
            metadata=metadata,
        )

    def _extract_comments(self, text: str, language: str) -> List[str]:
        """Extract comments from code."""
        comments = []
        patterns = COMMENT_PATTERNS.get(language, {})

        # Extract block comments first
        block_pattern = patterns.get("block", "")
        if block_pattern:
            for match in re.finditer(block_pattern, text, re.MULTILINE):
                comment = match.group(0)
                # Clean up block comment markers
                comment = self._clean_comment(comment, language)
                if comment.strip():
                    comments.append(comment.strip())

        # Extract line comments
        line_pattern = patterns.get("line", "")
        if line_pattern:
            for match in re.finditer(line_pattern, text, re.MULTILINE):
                comment = match.group(0)
                # Remove comment marker
                comment = re.sub(r"^(//|#|--)\s*", "", comment)
                if comment.strip():
                    comments.append(comment.strip())

        return comments

    def _clean_comment(self, comment: str, language: str) -> str:
        """Clean block comment markers."""
        if language == "python":
            # Remove triple quotes
            comment = re.sub(r'^"""|"""$|^\'\'\'|\'\'\'$', "", comment)
        elif language in ("javascript", "typescript", "java", "c", "cpp", "go", "rust"):
            # Remove /* */
            comment = re.sub(r"^/\*+|\*+/$", "", comment)
            # Remove leading * on lines
            comment = re.sub(r"^\s*\*\s?", "", comment, flags=re.MULTILINE)
        elif language == "ruby":
            comment = re.sub(r"^=begin|^=end", "", comment, flags=re.MULTILINE)
        elif language == "lua":
            comment = re.sub(r"^--\[\[|\]\]$", "", comment)

        return comment

    def _extract_structure(self, text: str, language: str) -> Optional[Dict]:
        """Extract code structure (functions, classes)."""
        structure = {"functions": [], "classes": []}

        if language == "python":
            # Find function definitions
            for match in re.finditer(r"^def\s+(\w+)\s*\(", text, re.MULTILINE):
                structure["functions"].append(match.group(1))

            # Find class definitions
            for match in re.finditer(r"^class\s+(\w+)", text, re.MULTILINE):
                structure["classes"].append(match.group(1))

        elif language in ("javascript", "typescript"):
            # Find function definitions
            for match in re.finditer(
                r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))",
                text,
            ):
                name = match.group(1) or match.group(2)
                if name:
                    structure["functions"].append(name)

            # Find class definitions
            for match in re.finditer(r"class\s+(\w+)", text):
                structure["classes"].append(match.group(1))

        elif language == "go":
            # Find function definitions
            for match in re.finditer(r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", text):
                structure["functions"].append(match.group(1))

            # Find struct definitions (Go's "classes")
            for match in re.finditer(r"type\s+(\w+)\s+struct", text):
                structure["classes"].append(match.group(1))

        elif language == "rust":
            # Find function definitions
            for match in re.finditer(r"fn\s+(\w+)\s*[<(]", text):
                structure["functions"].append(match.group(1))

            # Find struct/impl definitions
            for match in re.finditer(r"(?:struct|impl|enum)\s+(\w+)", text):
                structure["classes"].append(match.group(1))

        elif language == "java":
            # Find method definitions
            for match in re.finditer(
                r"(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\(",
                text,
            ):
                name = match.group(1)
                if name not in ("if", "while", "for", "switch"):
                    structure["functions"].append(name)

            # Find class definitions
            for match in re.finditer(r"class\s+(\w+)", text):
                structure["classes"].append(match.group(1))

        # Return None if no structure found
        if not structure["functions"] and not structure["classes"]:
            return None

        return structure
