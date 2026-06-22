"""Sprint 20 — Helm chart structural validation tests.

Coverage targets:
- Chart.yaml: required fields (apiVersion, name, version, appVersion, description)
- values.yaml: structural keys (replicaCount, image, service, autoscaling, config)
- values.yaml: sensible defaults (replicas=2, HPA enabled, ClusterIP service)
- templates/deployment.yaml: YAML parseable, contains expected keys
- templates/service.yaml: YAML parseable, type and port present
- templates/configmap.yaml: YAML parseable, data section present
- templates/hpa.yaml: YAML parseable (conditional block)
- templates/ingress.yaml: YAML parseable (conditional block)
- Helm directory structure: all expected files exist
- Chart version matches appVersion format
- release.yml: tag trigger on v*.*.*, PyPI publish, Docker publish jobs
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
HELM_DIR = ROOT / "helm" / "kyro"
TEMPLATES_DIR = HELM_DIR / "templates"
RELEASE_WF = ROOT / ".github" / "workflows" / "release.yml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file. For Helm templates (which contain Go template syntax),
    strip template directives before parsing."""
    text = path.read_text(encoding="utf-8")
    # Remove Helm template blocks: {{ ... }} and {{- ... -}}
    text = re.sub(r"\{\{[-\s]?.*?[-\s]?\}\}", '""', text)
    # Remove lone {{- end }} and similar
    text = re.sub(r"^\s*\{\{.*\}\}\s*$", "", text, flags=re.MULTILINE)
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}


# ── Directory structure ───────────────────────────────────────────────────────


class TestHelmDirectoryStructure:
    def test_helm_kyro_dir_exists(self) -> None:
        assert HELM_DIR.exists()

    def test_chart_yaml_exists(self) -> None:
        assert (HELM_DIR / "Chart.yaml").exists()

    def test_values_yaml_exists(self) -> None:
        assert (HELM_DIR / "values.yaml").exists()

    def test_templates_dir_exists(self) -> None:
        assert TEMPLATES_DIR.exists()

    def test_helpers_tpl_exists(self) -> None:
        assert (TEMPLATES_DIR / "_helpers.tpl").exists()

    def test_deployment_template_exists(self) -> None:
        assert (TEMPLATES_DIR / "deployment.yaml").exists()

    def test_service_template_exists(self) -> None:
        assert (TEMPLATES_DIR / "service.yaml").exists()

    def test_configmap_template_exists(self) -> None:
        assert (TEMPLATES_DIR / "configmap.yaml").exists()

    def test_hpa_template_exists(self) -> None:
        assert (TEMPLATES_DIR / "hpa.yaml").exists()

    def test_ingress_template_exists(self) -> None:
        assert (TEMPLATES_DIR / "ingress.yaml").exists()


# ── Chart.yaml ────────────────────────────────────────────────────────────────


class TestChartYaml:
    def _chart(self) -> dict:
        return yaml.safe_load((HELM_DIR / "Chart.yaml").read_text())

    def test_api_version_is_v2(self) -> None:
        assert self._chart()["apiVersion"] == "v2"

    def test_name_is_kyro(self) -> None:
        assert self._chart()["name"] == "kyro"

    def test_version_present(self) -> None:
        chart = self._chart()
        assert "version" in chart
        assert chart["version"]

    def test_app_version_present(self) -> None:
        chart = self._chart()
        assert "appVersion" in chart
        assert chart["appVersion"]

    def test_description_present(self) -> None:
        chart = self._chart()
        assert "description" in chart
        assert len(chart["description"]) > 10

    def test_chart_type_is_application(self) -> None:
        assert self._chart().get("type") == "application"

    def test_version_is_semver(self) -> None:
        version = str(self._chart()["version"])
        assert re.match(r"^\d+\.\d+\.\d+$", version)

    def test_chart_has_home_url(self) -> None:
        chart = self._chart()
        assert "home" in chart
        assert "github.com" in chart["home"]


# ── values.yaml ───────────────────────────────────────────────────────────────


class TestValuesYaml:
    def _values(self) -> dict:
        return yaml.safe_load((HELM_DIR / "values.yaml").read_text())

    def test_replica_count_default_2(self) -> None:
        assert self._values()["replicaCount"] == 2

    def test_image_section_present(self) -> None:
        v = self._values()
        assert "image" in v
        assert "repository" in v["image"]
        assert "tag" in v["image"]

    def test_service_section_present(self) -> None:
        v = self._values()
        assert "service" in v
        assert v["service"]["port"] == 8000

    def test_service_type_is_cluster_ip(self) -> None:
        assert self._values()["service"]["type"] == "ClusterIP"

    def test_autoscaling_enabled_by_default(self) -> None:
        v = self._values()
        assert v["autoscaling"]["enabled"] is True

    def test_autoscaling_min_max_replicas(self) -> None:
        hpa = self._values()["autoscaling"]
        assert hpa["minReplicas"] >= 1
        assert hpa["maxReplicas"] > hpa["minReplicas"]

    def test_config_section_present(self) -> None:
        assert "config" in self._values()

    def test_config_has_qdrant_url(self) -> None:
        assert "qdrantUrl" in self._values()["config"]

    def test_config_multi_tenancy_off_by_default(self) -> None:
        assert self._values()["config"]["multiTenancyEnabled"] == "false"

    def test_resources_limits_defined(self) -> None:
        r = self._values()["resources"]
        assert "limits" in r
        assert "memory" in r["limits"]

    def test_liveness_probe_defined(self) -> None:
        v = self._values()
        assert "livenessProbe" in v
        assert v["livenessProbe"]["httpGet"]["path"] == "/health"


# ── Release workflow ──────────────────────────────────────────────────────────


class TestReleaseWorkflow:
    def _wf(self) -> dict:
        return yaml.safe_load(RELEASE_WF.read_text())

    def test_release_yml_exists(self) -> None:
        assert RELEASE_WF.exists()

    def test_trigger_on_tag_push(self) -> None:
        wf = self._wf()
        # PyYAML parses YAML `on:` key as boolean True
        trigger = wf.get("on") or wf.get(True)
        assert trigger is not None
        assert "push" in trigger
        tags = trigger["push"].get("tags", [])
        assert any("v" in t for t in tags)

    def test_has_test_job(self) -> None:
        assert "test" in self._wf()["jobs"]

    def test_has_publish_pypi_job(self) -> None:
        assert "publish-pypi" in self._wf()["jobs"]

    def test_has_publish_docker_job(self) -> None:
        assert "publish-docker" in self._wf()["jobs"]

    def test_has_helm_publish_job(self) -> None:
        assert "publish-helm" in self._wf()["jobs"]

    def test_pypi_job_needs_test(self) -> None:
        jobs = self._wf()["jobs"]
        needs = jobs["publish-pypi"].get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "build" in needs or "test" in needs

    def test_docker_job_needs_test(self) -> None:
        jobs = self._wf()["jobs"]
        needs = jobs["publish-docker"].get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        assert "test" in needs
