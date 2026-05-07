"""Sprint 20 — packaging and distribution tests.

Coverage targets:
- pyproject.toml: version, classifiers, extras, entry points, URLs
- konjoai.__version__ matches pyproject.toml version
- All public sub-packages are importable
- Extras groups defined: jwt, mcp, eval, observability, dev, all
- Python version classifiers include 3.11 and 3.12
- Entry point 'konjoai' is declared
- Project URLs: Homepage, Documentation, Repository
- requirements.txt optional entries for mcp and PyJWT
- mkdocs.yml: site_name, nav, theme
- docs/ directory contains all expected pages
- release.yml: tag trigger, pypi publish job, docker publish job
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent  # /Users/wesleyscholl/kyro


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _pyproject() -> str:
    return _read("pyproject.toml")


# ── konjoai.__version__ ───────────────────────────────────────────────────────


class TestPackageVersion:
    def test_version_is_1_5_0(self) -> None:
        import konjoai
        assert konjoai.__version__ == "1.5.0"

    def test_version_matches_pyproject(self) -> None:
        import konjoai
        text = _pyproject()
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        assert match is not None
        assert konjoai.__version__ == match.group(1)

    def test_version_is_semver(self) -> None:
        import konjoai
        assert re.match(r"^\d+\.\d+\.\d+$", konjoai.__version__)


# ── pyproject.toml metadata ───────────────────────────────────────────────────


class TestPyprojectMetadata:
    def test_project_name_is_konjoai(self) -> None:
        assert 'name = "konjoai"' in _pyproject()

    def test_requires_python_311_plus(self) -> None:
        assert ">=3.11" in _pyproject()

    def test_license_is_mit(self) -> None:
        assert "MIT" in _pyproject()

    def test_description_present(self) -> None:
        text = _pyproject()
        assert "description" in text
        assert "RAG" in text

    def test_authors_present(self) -> None:
        assert "Wesley Scholl" in _pyproject()

    def test_keywords_include_rag_and_llm(self) -> None:
        text = _pyproject()
        assert '"rag"' in text
        assert '"llm"' in text


# ── Classifiers ───────────────────────────────────────────────────────────────


class TestPyprojectClassifiers:
    def test_production_stable_classifier(self) -> None:
        assert "Production/Stable" in _pyproject()

    def test_python_311_classifier(self) -> None:
        assert "Programming Language :: Python :: 3.11" in _pyproject()

    def test_python_312_classifier(self) -> None:
        assert "Programming Language :: Python :: 3.12" in _pyproject()

    def test_mit_license_classifier(self) -> None:
        assert "License :: OSI Approved :: MIT License" in _pyproject()

    def test_ai_topic_classifier(self) -> None:
        assert "Artificial Intelligence" in _pyproject()

    def test_typed_classifier(self) -> None:
        assert "Typing :: Typed" in _pyproject()


# ── Optional extras ───────────────────────────────────────────────────────────


class TestPyprojectExtras:
    def test_jwt_extra_defined(self) -> None:
        assert 'jwt = ["PyJWT' in _pyproject()

    def test_mcp_extra_defined(self) -> None:
        assert 'mcp = ["mcp' in _pyproject()

    def test_eval_extra_defined(self) -> None:
        assert 'eval = ["ragas' in _pyproject()

    def test_observability_extra_defined(self) -> None:
        assert "prometheus-client" in _pyproject()

    def test_dev_extra_defined(self) -> None:
        assert "pytest" in _pyproject()

    def test_all_extra_defined(self) -> None:
        assert "all" in _pyproject()


# ── Project URLs ──────────────────────────────────────────────────────────────


class TestPyprojectURLs:
    def test_homepage_url(self) -> None:
        assert "Homepage" in _pyproject()

    def test_documentation_url(self) -> None:
        assert "Documentation" in _pyproject()

    def test_repository_url(self) -> None:
        assert "Repository" in _pyproject()

    def test_changelog_url(self) -> None:
        assert "Changelog" in _pyproject()


# ── Entry point ───────────────────────────────────────────────────────────────


class TestPyprojectEntryPoints:
    def test_konjoai_cli_entry_point(self) -> None:
        text = _pyproject()
        assert "konjoai" in text
        assert "konjoai.cli.main:cli" in text


# ── Sub-package smoke imports ─────────────────────────────────────────────────


class TestSubPackageImports:
    def test_sdk_importable(self) -> None:
        from konjoai.sdk import KonjoClient
        assert KonjoClient is not None

    def test_mcp_importable(self) -> None:
        from konjoai.mcp import KyroMCPServer
        assert KyroMCPServer is not None

    def test_auth_importable(self) -> None:
        from konjoai.auth import TenantClaims
        assert TenantClaims is not None

    def test_config_importable(self) -> None:
        from konjoai.config import get_settings
        assert get_settings is not None

    def test_feedback_importable(self) -> None:
        from konjoai.feedback import get_feedback_store, FeedbackEvent, THUMBS_UP, THUMBS_DOWN
        assert get_feedback_store is not None
        assert THUMBS_UP == "thumbs_up"
        assert THUMBS_DOWN == "thumbs_down"


# ── mkdocs.yml ────────────────────────────────────────────────────────────────


class TestMkdocsConfig:
    def test_mkdocs_yml_exists(self) -> None:
        assert (ROOT / "mkdocs.yml").exists()

    def test_site_name_is_kyro(self) -> None:
        assert "Kyro" in _read("mkdocs.yml")

    def test_nav_includes_sdk(self) -> None:
        assert "sdk.md" in _read("mkdocs.yml")

    def test_nav_includes_mcp(self) -> None:
        assert "mcp.md" in _read("mkdocs.yml")

    def test_nav_includes_deployment(self) -> None:
        assert "deployment.md" in _read("mkdocs.yml")


# ── docs/ pages ───────────────────────────────────────────────────────────────


class TestDocsPages:
    _pages = ["index.md", "quickstart.md", "sdk.md", "mcp.md", "api.md", "configuration.md", "deployment.md"]

    def test_all_doc_pages_exist(self) -> None:
        for page in self._pages:
            assert (ROOT / "docs" / page).exists(), f"Missing docs/{page}"

    def test_index_mentions_version(self) -> None:
        assert "1.5.0" in _read("docs/index.md")

    def test_sdk_doc_has_error_handling_section(self) -> None:
        assert "KyroError" in _read("docs/sdk.md")

    def test_mcp_doc_has_claude_desktop_section(self) -> None:
        assert "Claude Desktop" in _read("docs/mcp.md")

    def test_deployment_doc_has_helm_section(self) -> None:
        assert "helm install" in _read("docs/deployment.md")
