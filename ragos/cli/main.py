from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger("ragos")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """RagOS — production RAG pipeline.

    \b
    Quick start:
        ragos ingest docs/
        ragos query "What is the refund policy?"
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--strategy", default="recursive", show_default=True, help="Chunking strategy: recursive | sentence_window.")
@click.option("--chunk-size", default=512, show_default=True, help="Target token count per chunk.")
@click.option("--overlap", default=64, show_default=True, help="Overlap between consecutive chunks.")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Suppress informational output.")
@click.pass_context
def ingest(ctx: click.Context, path: str, strategy: str, chunk_size: int, overlap: int, quiet: bool) -> None:
    """Ingest documents from PATH into the vector store.

    \b
    Examples:
        ragos ingest docs/
        ragos ingest README.md --strategy sentence_window
    """
    from ragos.ingest.loaders import load_path
    from ragos.ingest.chunkers import get_chunker
    from ragos.embed.encoder import get_encoder
    from ragos.store.qdrant import get_store
    from ragos.retrieve.sparse import get_sparse_index

    root = Path(path)
    chunker = get_chunker(strategy, chunk_size, overlap)
    encoder = get_encoder()
    store = get_store()

    all_contents: list[str] = []
    all_sources: list[str] = []
    all_metadatas: list[dict] = []
    sources_seen: set[str] = set()

    doc_count = 0
    for doc in load_path(root):
        doc_count += 1
        if not quiet:
            click.echo(f"  loading  {doc.source}", err=True)
        sources_seen.add(doc.source)
        for chunk in chunker.chunk(doc):
            all_contents.append(chunk.content)
            all_sources.append(chunk.source)
            all_metadatas.append(chunk.metadata)

    if not all_contents:
        click.echo("No content found at that path.", err=True)
        raise SystemExit(1)

    if not quiet:
        click.echo(f"Embedding {len(all_contents)} chunks…", err=True)

    embeddings = encoder.encode(all_contents)
    store.upsert(embeddings, all_contents, all_sources, all_metadatas)

    bm25 = get_sparse_index()
    bm25.build(all_contents, all_sources, all_metadatas)

    if not quiet:
        click.echo(
            f"✓ Indexed {len(all_contents)} chunks from {len(sources_seen)} source(s)."
        )


@cli.command()
@click.argument("question")
@click.option("--top-k", default=5, show_default=True, help="Number of final passages to include in context.")
@click.option("--quiet", "-q", is_flag=True, default=False, help="Print only the answer.")
@click.pass_context
def query(ctx: click.Context, question: str, top_k: int, quiet: bool) -> None:
    """Query the RAG pipeline with QUESTION.

    \b
    Examples:
        ragos query "What is the refund policy?"
        ragos query --top-k 10 "Summarize the architecture"
    """
    from ragos.retrieve.hybrid import hybrid_search
    from ragos.retrieve.reranker import rerank
    from ragos.generate.generator import get_generator

    hybrid_results = hybrid_search(question)
    reranked = rerank(question, hybrid_results, top_k=top_k)

    if not reranked:
        click.echo("No relevant documents found in the index. Run `ragos ingest` first.", err=True)
        raise SystemExit(1)

    context = "\n\n---\n\n".join(r.content for r in reranked)
    generator = get_generator()
    result = generator.generate(question=question, context=context)

    click.echo(result.answer)

    if not quiet:
        click.echo("\nSources:", err=True)
        seen: set[str] = set()
        for r in reranked:
            if r.source not in seen:
                click.echo(f"  • {r.source}  (score={r.score:.4f})", err=True)
                seen.add(r.source)


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--reload", is_flag=True, default=False, help="Enable hot-reload (dev only).")
@click.option("--quiet", "-q", is_flag=True, default=False)
def serve(host: str, port: int, reload: bool, quiet: bool) -> None:
    """Start the RagOS FastAPI server.

    \b
    Examples:
        ragos serve
        ragos serve --port 9000 --reload
    """
    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn is required: pip install uvicorn", err=True)
        raise SystemExit(2)

    if not quiet:
        click.echo(f"Starting RagOS server on http://{host}:{port} …")

    uvicorn.run(
        "ragos.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="warning" if quiet else "info",
    )


@cli.command()
@click.option("--quiet", "-q", is_flag=True, default=False)
def status(quiet: bool) -> None:
    """Show index status (vector count, BM25 built flag).

    \b
    Example:
        ragos status
    """
    from ragos.store.qdrant import get_store
    from ragos.retrieve.sparse import get_sparse_index

    store = get_store()
    bm25 = get_sparse_index()

    if quiet:
        click.echo(store.count())
    else:
        click.echo(f"Vector store: {store.count()} chunks indexed")
        click.echo(f"BM25 index:   {'ready' if bm25.built else 'not built (run ragos ingest)'}")


if __name__ == "__main__":
    cli()
