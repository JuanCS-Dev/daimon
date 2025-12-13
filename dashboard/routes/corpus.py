"""
DAIMON Dashboard - Corpus endpoints.

Includes: stats, tree, texts, search, upload, CRUD.
"""

from fastapi import APIRouter, Request

from ..models import CorpusTextCreate


router = APIRouter(prefix="/api/corpus", tags=["corpus"])


@router.get("/stats")
async def get_corpus_stats():
    """Estatisticas do corpus de sabedoria."""
    try:
        from corpus import CorpusManager
        manager = CorpusManager()
        return manager.get_stats()
    except ImportError:
        return {"error": "Corpus not available"}


@router.get("/tree")
async def get_corpus_tree():
    """Retorna arvore do corpus organizada por categoria."""
    try:
        from corpus import CorpusManager
        manager = CorpusManager()

        # Organizar por categoria
        tree: dict = {}
        for text in manager.texts.values():
            parts = text.category.split("/")
            current = tree
            for part in parts:
                if part not in current:
                    current[part] = {"_texts": []}
                current = current[part]
            current["_texts"].append({
                "id": text.id,
                "author": text.author,
                "title": text.title,
                "themes": text.themes,
                "relevance": text.relevance_score,
            })

        return {
            "tree": tree,
            "total": len(manager.texts),
            "categories": list(manager.list_categories().keys()),
            "authors": manager.list_authors(),
            "themes": manager.list_themes(),
        }
    except ImportError:
        return {"error": "Corpus not available"}


@router.get("/texts")
async def list_corpus_texts(category: str = ""):
    """Lista todos os textos do corpus, opcionalmente filtrado por categoria."""
    try:
        from corpus import CorpusManager
        manager = CorpusManager()

        if category:
            texts = manager.get_by_category(category)
        else:
            texts = list(manager.texts.values())

        return {
            "texts": [
                {
                    "id": t.id,
                    "author": t.author,
                    "title": t.title,
                    "category": t.category,
                    "content": t.content,
                    "themes": t.themes,
                    "relevance": t.relevance_score,
                    "source": t.source,
                }
                for t in texts
            ],
            "total": len(texts),
        }
    except ImportError:
        return {"error": "Corpus not available"}


@router.get("/text/{text_id}")
async def get_corpus_text(text_id: str):
    """Retorna um texto especifico do corpus."""
    try:
        from corpus import CorpusManager
        manager = CorpusManager()
        text = manager.get_text(text_id)

        if not text:
            return {"error": "Text not found"}

        return {
            "id": text.id,
            "author": text.author,
            "title": text.title,
            "category": text.category,
            "content": text.content,
            "themes": text.themes,
            "relevance": text.relevance_score,
            "source": text.source,
        }
    except ImportError:
        return {"error": "Corpus not available"}


@router.post("/text")
async def add_corpus_text(data: CorpusTextCreate):
    """Adiciona novo texto ao corpus."""
    try:
        from corpus import CorpusManager, TextMetadata
        manager = CorpusManager()

        # Verificar categoria valida
        valid_categories = list(manager.list_categories().keys())
        if data.category not in valid_categories:
            return {
                "error": f"Invalid category. Valid: {valid_categories}"
            }

        metadata = TextMetadata(
            source=data.source,
            relevance_score=data.relevance,
            themes=data.themes,
        )

        text_id = manager.add_text(
            author=data.author,
            title=data.title,
            category=data.category,
            content=data.content,
            metadata=metadata,
        )

        return {
            "status": "created",
            "id": text_id,
            "category": data.category,
        }
    except ImportError:
        return {"error": "Corpus not available"}
    except Exception as e:
        return {"error": str(e)}


@router.delete("/text/{text_id}")
async def delete_corpus_text(text_id: str):
    """Remove texto do corpus."""
    try:
        from corpus import CorpusManager
        manager = CorpusManager()

        if manager.delete_text(text_id):
            return {"status": "deleted", "id": text_id}
        else:
            return {"error": "Text not found"}
    except ImportError:
        return {"error": "Corpus not available"}


@router.post("/upload")
async def upload_corpus_file(request: Request):
    """Upload arquivo para o corpus (TXT, MD, PDF)."""
    try:
        from corpus import CorpusManager, TextMetadata

        form = await request.form()
        file = form.get("file")
        title = form.get("title", "")
        author = form.get("author", "Unknown")
        category = form.get("category", "filosofia/geral")
        themes = form.get("themes", "")
        source = form.get("source", "")
        relevance = float(form.get("relevance", 0.5))

        if not file:
            return {"error": "No file provided"}

        # Read file content
        content = await file.read()
        filename = file.filename

        # Detect file type and process
        if filename.endswith(".pdf"):
            try:
                from corpus.loaders import PDFLoader
                loader = PDFLoader()
                import tempfile
                from pathlib import Path
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                doc = loader.load(Path(tmp_path))
                text_content = doc.content
                Path(tmp_path).unlink()
            except ImportError:
                return {"error": "PDF support not available (install pypdf)"}
        else:
            text_content = content.decode("utf-8")

        if not title:
            title = filename.rsplit(".", 1)[0]

        manager = CorpusManager()

        metadata = TextMetadata(
            source=source or filename,
            relevance_score=relevance,
            themes=[t.strip() for t in themes.split(",") if t.strip()],
        )

        text_id = manager.add_text(
            author=author,
            title=title,
            category=category,
            content=text_content,
            metadata=metadata,
        )

        return {
            "status": "uploaded",
            "id": text_id,
            "title": title,
            "size": len(text_content),
        }
    except ImportError:
        return {"error": "Corpus not available"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/search")
async def search_corpus(q: str = "", limit: int = 10):
    """Busca no corpus de sabedoria."""
    if not q:
        return {"results": [], "query": ""}

    try:
        from corpus import CorpusManager
        manager = CorpusManager()
        results = manager.search(q, limit=limit)
        return {
            "query": q,
            "results": [
                {
                    "id": t.id,
                    "author": t.author,
                    "title": t.title,
                    "category": t.category,
                    "content": t.content[:200] + "..." if len(t.content) > 200 else t.content,
                    "themes": t.themes,
                }
                for t in results
            ],
        }
    except ImportError:
        return {"error": "Corpus not available"}
