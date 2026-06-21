"""Lightweight GraphRAG retriever using NetworkX community detection.

Sprint 15 contract (v0.8.5):
1. Build a token-overlap similarity graph from hybrid search results (Jaccard edges).
2. Detect communities using ``greedy_modularity_communities`` (Louvain-style).
3. Select the top-scoring chunk (by RRF score) from each community as representative.
4. Return ranked CommunityContext objects + a flat representative list for downstream reranking.
5. Feature-flagged via ``Settings.enable_graph_rag`` (K3: transparent pass-through when disabled).
6. Graceful fallback when networkx is unavailable — returns raw hybrid results (K5).

K1 — all exceptions re-raised with ``from exc`` chain.
K2 — logger.warning on every error/fallback path.
K5 — networkx is the only new dependency; import is optional.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities

    _HAS_NETWORKX = True
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]
    greedy_modularity_communities = None  # type: ignore[assignment]
    _HAS_NETWORKX = False


# ── Domain types ─────────────────────────────────────────────────────────────


@dataclass
class CommunityContext:
    """A semantically coherent community of retrieved chunks.

    Members are ordered by descending RRF score (best first).
    ``label`` is the first 80 characters of the best-scoring member and is
    suitable for display or telemetry without further processing.
    """

    community_id: int
    members: list[str] = field(default_factory=list)  # chunk content, best-first
    sources: list[str] = field(default_factory=list)  # corresponding source paths
    rrf_scores: list[float] = field(default_factory=list)  # per-member RRF scores
    label: str = ""  # short human-readable label
    size: int = 0  # total community size


@dataclass
class GraphRAGResult:
    """Output of ``GraphRAGRetriever.retrieve``.

    ``representative_chunks`` is a drop-in replacement for ``hybrid_results``
    in the query pipeline — one best chunk per community, ordered by community
    relevance (highest intra-community RRF score first).

    ``used_fallback`` is True when networkx is absent or graph construction
    failed; in that case ``representative_chunks`` == the original input list.
    """

    communities: list[CommunityContext]
    community_labels: list[str]  # flat label list for QueryResponse field
    representative_chunks: list[Any]  # best chunk per community; replaces hybrid_results
    n_nodes: int
    n_edges: int
    used_fallback: bool


# ── Internal helpers ─────────────────────────────────────────────────────────


def _tokenize(text: str) -> frozenset[str]:
    """Return a frozenset of lowercase alpha tokens of length ≥ 3.

    Using frozenset (rather than Counter/set) makes Jaccard O(intersection)
    without any extra data structure overhead.
    """
    return frozenset(re.findall(r"\b[a-z]{3,}\b", text.lower()))


# ── Core components ───────────────────────────────────────────────────────────


class EntityGraph:
    """Builds an undirected similarity graph from chunk text via Jaccard overlap.

    Each node is a chunk index (0 … n-1).  An undirected edge is inserted
    between nodes *i* and *j* when::

        jaccard(tokens_i, tokens_j) >= similarity_threshold

    This approach requires only stdlib ``re`` — no additional embedding step,
    no GPU, no heavy dependencies.

    Parameters
    ----------
    similarity_threshold:
        Minimum Jaccard similarity [0, 1] required to create an edge.
        Lower values → more edges → coarser communities.
        Default 0.3 works well for short (512-token) RAG chunks.
    """

    def __init__(self, similarity_threshold: float = 0.3) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be in [0, 1]; got {similarity_threshold!r}")
        self.similarity_threshold = similarity_threshold

    def build(self, contents: list[str]) -> nx.Graph:
        """Build and return a NetworkX Graph from a list of chunk strings.

        Raises
        ------
        RuntimeError
            When networkx is not installed.
        """
        if not _HAS_NETWORKX:
            raise RuntimeError(
                "networkx is required for EntityGraph.build — install it with: pip install networkx>=3.2"
            )
        graph = nx.Graph()

        for idx, content in enumerate(contents):
            graph.add_node(idx, content=content)

        token_sets = [_tokenize(c) for c in contents]
        n = len(token_sets)

        for i in range(n):
            for j in range(i + 1, n):
                union = token_sets[i] | token_sets[j]
                if not union:
                    continue
                jaccard = len(token_sets[i] & token_sets[j]) / len(union)
                if jaccard >= self.similarity_threshold:
                    graph.add_edge(i, j, weight=jaccard)

        return graph

    def detect_communities(self, graph: nx.Graph) -> list[frozenset[int]]:
        """Return one frozenset of node ids per detected community.

        Uses ``greedy_modularity_communities`` (Louvain-style greedy algorithm
        from NetworkX).  When the graph has no edges every node forms its own
        singleton community — this preserves all chunks rather than collapsing
        everything into one bucket.

        Raises
        ------
        RuntimeError
            When networkx is not installed.
        """
        if not _HAS_NETWORKX:
            raise RuntimeError(
                "networkx is required for community detection — install it with: pip install networkx>=3.2"
            )
        if graph.number_of_nodes() == 0:
            return []
        if graph.number_of_edges() == 0:
            # No edges: each node is its own singleton community.
            return [frozenset({n}) for n in graph.nodes()]
        return list(greedy_modularity_communities(graph, weight="weight"))


# ── Retriever ─────────────────────────────────────────────────────────────────


class GraphRAGRetriever:
    """Groups hybrid search results into semantic communities and returns representatives.

    Algorithm
    ---------
    1. Build a Jaccard-weighted graph over the hybrid result set.
    2. Detect communities with Louvain-style greedy modularity.
    3. Sort communities by their best-member RRF score (descending).
    4. Take the top ``max_communities`` communities.
    5. From each community return the highest-RRF-score chunk as representative.
    6. ``GraphRAGResult.representative_chunks`` replaces ``hybrid_results``
       in the query pipeline — the downstream reranker sees one chunk per
       semantic cluster rather than potentially many near-duplicates.

    Usage::

        retriever = GraphRAGRetriever(max_communities=5, similarity_threshold=0.3)
        result = retriever.retrieve(hybrid_results)
        # downstream:
        hybrid_results = result.representative_chunks
        query_response.graph_rag_communities = result.community_labels
    """

    def __init__(
        self,
        max_communities: int = 5,
        similarity_threshold: float = 0.3,
    ) -> None:
        if max_communities < 1:
            raise ValueError(f"max_communities must be ≥ 1; got {max_communities!r}")
        self.max_communities = max_communities
        self.entity_graph = EntityGraph(similarity_threshold=similarity_threshold)

    def retrieve(self, hybrid_results: list[Any]) -> GraphRAGResult:
        """Build community graph and return CommunityContext list plus representatives.

        Parameters
        ----------
        hybrid_results:
            Ordered list of HybridResult (or any object with ``.content``,
            ``.source``, ``.rrf_score`` attributes).

        Returns
        -------
        GraphRAGResult
            ``representative_chunks`` is ready to replace ``hybrid_results``
            in the query pipeline.  ``community_labels`` is ready to attach to
            ``QueryResponse.graph_rag_communities``.
        """
        if not hybrid_results:
            return GraphRAGResult(
                communities=[],
                community_labels=[],
                representative_chunks=[],
                n_nodes=0,
                n_edges=0,
                used_fallback=False,
            )

        if not _HAS_NETWORKX:
            logger.warning(
                "GraphRAGRetriever: networkx not installed — falling back to raw hybrid results (install networkx>=3.2)"
            )
            return GraphRAGResult(
                communities=[],
                community_labels=[],
                representative_chunks=list(hybrid_results),
                n_nodes=len(hybrid_results),
                n_edges=0,
                used_fallback=True,
            )

        contents = [r.content for r in hybrid_results]

        try:
            graph = self.entity_graph.build(contents)
            raw_communities = self.entity_graph.detect_communities(graph)
        except Exception as exc:
            logger.warning(
                "GraphRAGRetriever: graph construction failed (%s) — falling back to raw hybrid results",
                exc,
            )
            return GraphRAGResult(
                communities=[],
                community_labels=[],
                representative_chunks=list(hybrid_results),
                n_nodes=len(hybrid_results),
                n_edges=0,
                used_fallback=True,
            )

        def _best_score(node_ids: frozenset[int]) -> float:
            return max(hybrid_results[idx].rrf_score for idx in node_ids)

        sorted_communities = sorted(raw_communities, key=_best_score, reverse=True)
        top_communities = sorted_communities[: self.max_communities]

        community_contexts: list[CommunityContext] = []
        representatives: list[Any] = []

        for cid, node_ids in enumerate(top_communities):
            members_sorted = sorted(
                node_ids,
                key=lambda idx: hybrid_results[idx].rrf_score,
                reverse=True,
            )
            best_member = hybrid_results[members_sorted[0]]

            members_content = [hybrid_results[idx].content for idx in members_sorted]
            members_sources = [hybrid_results[idx].source for idx in members_sorted]
            members_scores = [hybrid_results[idx].rrf_score for idx in members_sorted]

            # Label: first 80 chars of the best-scoring member (stripped)
            label = best_member.content[:80].rstrip()

            community_contexts.append(
                CommunityContext(
                    community_id=cid,
                    members=members_content,
                    sources=members_sources,
                    rrf_scores=members_scores,
                    label=label,
                    size=len(members_sorted),
                )
            )
            representatives.append(best_member)

        community_labels = [ctx.label for ctx in community_contexts]

        return GraphRAGResult(
            communities=community_contexts,
            community_labels=community_labels,
            representative_chunks=representatives,
            n_nodes=graph.number_of_nodes(),
            n_edges=graph.number_of_edges(),
            used_fallback=False,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_graph_rag_retriever: GraphRAGRetriever | None = None


def get_graph_rag_retriever(
    max_communities: int = 5,
    similarity_threshold: float = 0.3,
) -> GraphRAGRetriever:
    """Return the module-level GraphRAGRetriever singleton (lazy init).

    On first call the singleton is created with the supplied parameters.
    Subsequent calls return the cached instance regardless of parameters —
    consistent with the CRAG and Self-RAG singleton pattern in this codebase.
    """
    global _graph_rag_retriever  # noqa: PLW0603
    if _graph_rag_retriever is None:
        _graph_rag_retriever = GraphRAGRetriever(
            max_communities=max_communities,
            similarity_threshold=similarity_threshold,
        )
    return _graph_rag_retriever
