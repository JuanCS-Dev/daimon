# DAIMON Corpus

**Sistema de Corpus - Textos de Sabedoria para Embasamento Ético**

---

## Visão Geral

O Corpus é uma coleção curada de textos filosóficos e de sabedoria que fundamentam o raciocínio ético do DAIMON. Inspirado na tradição filosófica clássica, fornece referências para reflexão e julgamento.

### Propósito

1. **Embasamento Ético** - Textos clássicos para fundamentar decisões
2. **Referência Semântica** - Busca por conceitos e temas
3. **Contexto Cultural** - Sabedoria acumulada pela humanidade
4. **Ancoragem** - Evitar deriva moral em sistemas de IA

---

## Arquitetura

```
corpus/
├── __init__.py              # Exports principais
├── manager.py               # CorpusManager - gestão central
├── models.py                # WisdomText, TextMetadata
├── search.py                # Busca por keywords
├── semantic_search.py       # Busca semântica (FAISS)
├── bootstrap_texts.py       # Textos fundacionais
└── loaders/
    ├── __init__.py
    ├── base.py              # BaseLoader abstrato
    ├── text.py              # TextLoader, MarkdownLoader
    ├── pdf.py               # PDFLoader
    ├── code.py              # CodeLoader
    └── web.py               # WebLoader
```

### Storage

```
~/.daimon/corpus/
├── filosofia/
│   ├── gregos/
│   │   ├── aristotle_nicomachean_ethics_on_t.json
│   │   ├── plato_republic_allegory_of_the_ca.json
│   │   └── socrates_apology_on_wisdom.json
│   └── estoicos/
│       ├── marcus_aurelius_meditations_on_mo.json
│       ├── epictetus_enchiridion_on_control.json
│       └── seneca_letters_on_the_shortness_o.json
├── teologia/
├── ciencia/
│   ├── karl_popper_logic_of_scientific_disc.json
│   └── richard_feynman_on_scientific_method.json
├── logica/
│   └── aristotle_organon_on_valid_reasoning.json
├── etica/
│   └── immanuel_kant_groundwork_categorical_.json
├── literatura/
├── _index/
│   ├── by_theme.json
│   └── by_author.json
└── _semantic/
    ├── vectors.faiss
    └── documents.json
```

---

## CorpusManager

**Arquivo:** `corpus/manager.py`
**Status:** ✅ Funcional

O CorpusManager é a interface principal para gerenciar textos de sabedoria.

### Categorias Disponíveis

| Categoria | Descrição |
|-----------|-----------|
| `filosofia/gregos` | Filosofia Grega Clássica |
| `filosofia/estoicos` | Filosofia Estoica |
| `filosofia/modernos` | Filosofia Moderna |
| `teologia` | Teologia e Religião |
| `ciencia` | Ciência e Método |
| `logica` | Lógica e Raciocínio |
| `etica` | Ética e Moralidade |
| `literatura` | Literatura e Poesia |

### API

```python
from corpus import CorpusManager, TextMetadata

manager = CorpusManager()

# Adicionar texto
text_id = manager.add_text(
    author="Epictetus",
    title="Enchiridion - On Freedom",
    category="filosofia/estoicos",
    content="Make the best use of what is in your power...",
    metadata=TextMetadata(
        source="Enchiridion, Chapter 1",
        themes=["freedom", "control", "wisdom"],
        relevance_score=0.9,
    ),
)
# → "epictetus_enchiridion_on_freedom"

# Buscar por keywords
results = manager.search("virtue", limit=5)
# → [WisdomText(...), ...]

# Buscar por semântica (se disponível)
results = manager.semantic_search("what is true wisdom?", limit=5)
# → [WisdomText(...), ...]

# Buscar híbrida (keyword + semântica)
results = manager.hybrid_search("ethics and duty", limit=5, semantic_weight=0.7)
# → [WisdomText(...), ...]

# Por autor
texts = manager.get_by_author("Aristotle")

# Por tema
texts = manager.get_by_theme("virtue")

# Por categoria
texts = manager.get_by_category("filosofia/estoicos")

# Estatísticas
stats = manager.get_stats()
# → {
#     "total_texts": 10,
#     "total_authors": 7,
#     "total_themes": 25,
#     "by_category": {"filosofia/estoicos": 3, ...},
#     "semantic_enabled": True,
#     "semantic_indexed": 10,
# }

# Deletar texto
manager.delete_text("epictetus_enchiridion_on_freedom")

# Reindexar semântica
count = manager.reindex_semantic()
```

---

## Modelos de Dados

### WisdomText

```python
@dataclass
class WisdomText:
    id: str              # ID único (autor_titulo)
    author: str          # Nome do autor
    title: str           # Título do texto
    category: str        # Categoria (ex: "filosofia/estoicos")
    content: str         # Conteúdo do texto
    themes: List[str]    # Temas (ex: ["virtue", "wisdom"])
    metadata: TextMetadata

    @property
    def source(self) -> str
    @property
    def added_at(self) -> str
    @property
    def relevance_score(self) -> float
```

### TextMetadata

```python
@dataclass
class TextMetadata:
    source: str = ""           # Referência original
    added_at: str = ""         # ISO timestamp
    relevance_score: float = 0.5  # 0.0-1.0
    themes: List[str] = []     # Temas do texto
```

---

## Busca

### Keyword Search

**Arquivo:** `corpus/search.py`

Busca simples por substring em conteúdo, título e temas.

```python
from corpus import keyword_search

# Retorna lista de WisdomText
results = keyword_search(manager.texts, "virtue", limit=10)

# Ou via manager
results = manager.search("virtue", limit=10)
```

**Scoring:**
- Match no título: +3 pontos
- Match no conteúdo: +2 pontos
- Match em tema: +2 pontos

### Semantic Search

**Arquivo:** `corpus/semantic_search.py`
**Dependências:** `sentence-transformers`, `faiss-cpu`

Busca por similaridade semântica usando embeddings.

```python
from corpus import SemanticCorpus, get_semantic_corpus

corpus = get_semantic_corpus()

# Adicionar documento
doc_id = corpus.add_document(
    text="Virtue is knowledge.",
    metadata={"author": "Socrates"},
    doc_id="socrates_virtue",
)

# Buscar
results = corpus.search("what is wisdom?", k=5, min_score=0.3)
# → [SearchResult(text=..., score=0.85, metadata=..., doc_id=...), ...]

# Salvar índice
corpus.save()

# Estatísticas
stats = corpus.get_stats()
# → {"document_count": 10, "model_name": "all-MiniLM-L6-v2", ...}
```

**Modelo:** `all-MiniLM-L6-v2` (~90MB)
- Dimensão: 384
- Velocidade: ~50ms por embedding
- Busca: <10ms para 10k documentos

### Hybrid Search

Combina keyword e semântica com peso ajustável.

```python
from corpus import hybrid_search

results = hybrid_search(
    texts=manager.texts,
    semantic=corpus,
    query="ethics and duty",
    limit=5,
    semantic_weight=0.7,  # 70% semântica, 30% keyword
)
```

---

## Loaders

Os loaders extraem texto de diferentes formatos de arquivo.

### BaseLoader

```python
from corpus.loaders import BaseLoader, Document

class CustomLoader(BaseLoader):
    def load(self, path: str) -> Document:
        text = ...  # Extract text
        return Document(
            text=text,
            source=path,
            title="...",
            metadata={...},
        )
```

### TextLoader

**Arquivo:** `corpus/loaders/text.py`

Carrega arquivos `.txt` e `.rst`.

```python
from corpus.loaders import TextLoader

loader = TextLoader()
doc = loader.load("/path/to/file.txt")
```

### MarkdownLoader

Carrega arquivos `.md` com opção de strip de formatação.

```python
from corpus.loaders import MarkdownLoader

loader = MarkdownLoader(strip_formatting=True)
doc = loader.load("/path/to/README.md")
```

**Strip Features:**
- Remove headers (#)
- Remove bold/italic
- Remove links [text](url)
- Remove code blocks
- Remove blockquotes

### PDFLoader

**Arquivo:** `corpus/loaders/pdf.py`
**Dependência:** `pypdf`

Extrai texto de arquivos PDF.

```python
from corpus.loaders import PDFLoader

loader = PDFLoader(extract_metadata=True)
doc = loader.load("/path/to/book.pdf")

# Metadata extraída
doc.metadata
# → {"page_count": 100, "author": "...", "pdf_title": "..."}
```

### CodeLoader

**Arquivo:** `corpus/loaders/code.py`

Carrega código fonte com detecção de linguagem.

```python
from corpus.loaders import CodeLoader

loader = CodeLoader(
    extract_comments=True,  # Extrair comentários
    strip_comments=False,   # Manter código completo
)
doc = loader.load("/path/to/script.py")

# Metadata
doc.metadata["language"]  # "python"
doc.metadata["structure"]["functions"]  # ["main", "helper"]
doc.metadata["structure"]["classes"]  # ["MyClass"]
```

**Linguagens Suportadas:**
- Python, JavaScript, TypeScript
- Go, Rust, Java, C/C++
- Ruby, PHP, Lua
- Bash, SQL, Vim, Elisp

### WebLoader

**Arquivo:** `corpus/loaders/web.py`
**Dependências:** `httpx`, `beautifulsoup4`

Extrai texto de URLs web.

```python
from corpus.loaders import WebLoader

loader = WebLoader(timeout=30.0)
doc = loader.load("https://example.com/article")

doc.metadata["domain"]  # "example.com"
doc.metadata["description"]  # Meta description
```

**Features:**
- Remove scripts/styles/nav/footer
- Detecta área de conteúdo principal
- Extrai meta tags

---

## Bootstrap Texts

**Arquivo:** `corpus/bootstrap_texts.py`

Textos fundacionais pré-configurados para iniciar o corpus.

### Textos Incluídos

| Autor | Título | Categoria | Temas |
|-------|--------|-----------|-------|
| Marcus Aurelius | Meditations - On Morning | estoicos | virtue, stoicism |
| Epictetus | Enchiridion - On Control | estoicos | control, freedom |
| Seneca | Letters - Shortness of Life | estoicos | time, mortality |
| Aristotle | Nicomachean Ethics | gregos | ethics, virtue |
| Plato | Republic - Cave Allegory | gregos | knowledge, truth |
| Socrates | Apology - On Wisdom | gregos | wisdom, humility |
| Aristotle | Organon - Valid Reasoning | logica | logic, syllogism |
| Immanuel Kant | Categorical Imperative | etica | duty, universality |
| Karl Popper | Falsifiability | ciencia | science, method |
| Richard Feynman | Scientific Method | ciencia | honesty, integrity |

### Uso

```python
from corpus.bootstrap_texts import bootstrap_corpus

result = bootstrap_corpus()
# → {
#     "added": 10,
#     "skipped": 0,
#     "total": 10,
#     "corpus_stats": {...},
# }
```

O bootstrap é idempotente - textos já existentes são ignorados.

---

## Integração com DAIMON

### Com ReflectionEngine

O corpus pode ser consultado durante reflexões para embasar insights.

```python
# Exemplo de uso em reflexão
manager = CorpusManager()
relevant = manager.hybrid_search("patience and consistency", limit=3)
for text in relevant:
    logger.info("Reference: %s - %s", text.author, text.title)
```

### Com MCP Server

O corpus é acessível via MCP tools:

```python
# Tool: corpus_search
results = corpus_search(query="virtue", limit=5)
```

### Com Dashboard

Endpoints disponíveis:

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/api/corpus/stats` | GET | Estatísticas do corpus |
| `/api/corpus/search` | GET | Buscar textos (`?q=query`) |
| `/api/corpus/texts` | GET | Listar textos (`?category=`) |
| `/api/corpus/text/{id}` | GET | Obter texto específico |
| `/api/corpus/text` | POST | Adicionar texto |
| `/api/corpus/text/{id}` | DELETE | Remover texto |

---

## Performance

### Benchmarks

| Operação | Tempo | Notas |
|----------|-------|-------|
| Keyword search | <5ms | 1000 textos |
| Semantic search | <10ms | Após warm-up |
| Model loading | ~2s | Uma vez por sessão |
| Add document | ~50ms | Com embedding |
| Bootstrap | ~1s | 10 textos |

### Tamanhos

| Componente | Tamanho | Notas |
|------------|---------|-------|
| Modelo SBERT | ~90MB | all-MiniLM-L6-v2 |
| FAISS index | ~1KB/doc | Vetores 384-dim |
| Text JSON | ~1-5KB | Por texto |

---

## Testes

```bash
# Todos os testes de corpus
python -m pytest tests/test_corpus_manager.py -v

# Testar bootstrap
python corpus/bootstrap_texts.py

# Testar semantic search
python corpus/semantic_search.py
```

---

## Instalação de Dependências

```bash
# Core (já incluídas)
pip install numpy

# Semantic search (opcional)
pip install sentence-transformers faiss-cpu

# PDF loader (opcional)
pip install pypdf

# Web loader (opcional)
pip install httpx beautifulsoup4
```

---

## Troubleshooting

### Semantic search não funciona

```bash
# Verificar instalação
pip install sentence-transformers faiss-cpu

# Testar import
python -c "from sentence_transformers import SentenceTransformer; print('OK')"
python -c "import faiss; print('OK')"
```

### Bootstrap não encontra textos

```bash
# Verificar diretório
ls -la ~/.daimon/corpus/

# Recriar estrutura
python -c "from corpus import CorpusManager; CorpusManager()"
```

### Modelo não carrega

```bash
# Verificar cache
ls -la ~/.cache/torch/sentence_transformers/

# Forçar download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

---

## Limitações Honestas

1. **Corpus pequeno** - Apenas 10 textos bootstrap, expansão manual necessária
2. **Sem validação** - Textos adicionados não são verificados
3. **Embedding local** - Qualidade inferior a modelos maiores (GPT, Claude)
4. **Sem multilingual** - Modelo focado em inglês
5. **FAISS básico** - IndexFlatIP sem otimização para grandes volumes
6. **Sem deduplicação** - Textos similares não são detectados

---

## Filosofia do Sistema

> "The unexamined life is not worth living." - Socrates

O corpus não pretende substituir o julgamento humano. Ele oferece:

1. **Ancoragem** - Referências estáveis em meio a mudanças
2. **Perspectiva** - Sabedoria acumulada por milênios
3. **Humildade** - Reconhecer que não sabemos tudo
4. **Reflexão** - Pausar antes de agir

---

*Documentação atualizada em 2025-12-13*
