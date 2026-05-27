"""HTTP client mirroring the studio's REST API."""

from __future__ import annotations

import builtins
import time
from collections.abc import Mapping
from typing import Any, cast

import httpx

from oas.types import (
    Dataset,
    DatasetVersion,
    Experiment,
    Job,
    Model,
    ModelVersion,
    Project,
)

_TERMINAL = {"succeeded", "failed", "canceled"}


class _Resource:
    def __init__(self, http: httpx.Client) -> None:
        self._http = http


class ProjectsAPI(_Resource):
    def list(self) -> list[Project]:
        return [Project.model_validate(x) for x in self._http.get("/projects").raise_for_status().json()]

    def create(self, slug: str, name: str, **kwargs: Any) -> Project:
        r = self._http.post("/projects", json={"slug": slug, "name": name, **kwargs})
        r.raise_for_status()
        return Project.model_validate(r.json())

    def get(self, project_id: str) -> Project:
        return Project.model_validate(self._http.get(f"/projects/{project_id}").raise_for_status().json())

    def delete(self, project_id: str) -> None:
        self._http.delete(f"/projects/{project_id}").raise_for_status()


class DatasetsAPI(_Resource):
    def list(self, project_id: str | None = None, modality: str | None = None) -> list[Dataset]:
        params = {k: v for k, v in {"project_id": project_id, "modality": modality}.items() if v}
        r = self._http.get("/datasets", params=params).raise_for_status()
        return [Dataset.model_validate(x) for x in r.json()]

    def create(self, project_id: str, slug: str, name: str, modality: str, **kwargs: Any) -> Dataset:
        r = self._http.post(
            "/datasets",
            json={"project_id": project_id, "slug": slug, "name": name, "modality": modality, **kwargs},
        )
        r.raise_for_status()
        return Dataset.model_validate(r.json())

    def get(self, dataset_id: str) -> Dataset:
        return Dataset.model_validate(self._http.get(f"/datasets/{dataset_id}").raise_for_status().json())

    # NOTE: `builtins.list[...]` is used here because `list` is shadowed by the
    # `DatasetsAPI.list` method defined above; mypy resolves bare `list` in the
    # class scope to the method, producing a [valid-type] error.
    def versions(self, dataset_id: str) -> builtins.list[DatasetVersion]:
        r = self._http.get(f"/datasets/{dataset_id}/versions").raise_for_status()
        return [DatasetVersion.model_validate(x) for x in r.json()]

    def add_version(
        self,
        dataset_id: str,
        version: str,
        manifest_uri: str,
        **kwargs: Any,
    ) -> DatasetVersion:
        r = self._http.post(
            f"/datasets/{dataset_id}/versions",
            json={"version": version, "manifest_uri": manifest_uri, **kwargs},
        )
        r.raise_for_status()
        return DatasetVersion.model_validate(r.json())


class JobsAPI(_Resource):
    def kinds(self) -> list[str]:
        return cast(list[str], self._http.get("/jobs/handlers").raise_for_status().json())

    def list(self, project_id: str | None = None) -> list[Job]:
        params = {"project_id": project_id} if project_id else {}
        r = self._http.get("/jobs", params=params).raise_for_status()
        return [Job.model_validate(x) for x in r.json()]

    def submit(
        self,
        project_id: str,
        kind: str,
        name: str,
        config: Mapping[str, Any] | None = None,
        priority: int = 0,
    ) -> Job:
        r = self._http.post(
            "/jobs",
            json={
                "project_id": project_id,
                "kind": kind,
                "name": name,
                "config": dict(config or {}),
                "priority": priority,
            },
        )
        r.raise_for_status()
        return Job.model_validate(r.json())

    def get(self, job_id: str) -> Job:
        return Job.model_validate(self._http.get(f"/jobs/{job_id}").raise_for_status().json())

    def cancel(self, job_id: str) -> Job:
        r = self._http.post(f"/jobs/{job_id}/cancel").raise_for_status()
        return Job.model_validate(r.json())

    def wait(self, job_id: str, timeout: float = 600.0, poll_interval: float = 1.0) -> Job:
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = self.get(job_id)
            if j.status in _TERMINAL:
                return j
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")


class ExperimentsAPI(_Resource):
    def list(self, project_id: str | None = None) -> list[Experiment]:
        params = {"project_id": project_id} if project_id else {}
        r = self._http.get("/experiments", params=params).raise_for_status()
        return [Experiment.model_validate(x) for x in r.json()]

    def create(self, project_id: str, name: str, **kwargs: Any) -> Experiment:
        r = self._http.post("/experiments", json={"project_id": project_id, "name": name, **kwargs})
        r.raise_for_status()
        return Experiment.model_validate(r.json())


class ModelsAPI(_Resource):
    def list(self, project_id: str, modality: str | None = None) -> list[Model]:
        params: dict[str, Any] = {"project_id": project_id}
        if modality:
            params["modality"] = modality
        r = self._http.get("/models", params=params).raise_for_status()
        return [Model.model_validate(x) for x in r.json()]

    def create(self, project_id: str, slug: str, name: str, modality: str, **kwargs: Any) -> Model:
        r = self._http.post(
            "/models",
            json={"project_id": project_id, "slug": slug, "name": name, "modality": modality, **kwargs},
        )
        r.raise_for_status()
        return Model.model_validate(r.json())

    # See DatasetsAPI.versions: `builtins.list` avoids the `ModelsAPI.list`
    # method shadow.
    def versions(self, model_id: str) -> builtins.list[ModelVersion]:
        r = self._http.get(f"/models/{model_id}/versions").raise_for_status()
        return [ModelVersion.model_validate(x) for x in r.json()]

    def publish(self, model_id: str, version: str, artifact_uri: str, **kwargs: Any) -> ModelVersion:
        r = self._http.post(
            f"/models/{model_id}/versions",
            json={"version": version, "artifact_uri": artifact_uri, **kwargs},
        )
        r.raise_for_status()
        return ModelVersion.model_validate(r.json())

    def set_stage(self, version_id: str, stage: str) -> ModelVersion:
        r = self._http.post(f"/models/versions/{version_id}/stage", params={"stage": stage})
        r.raise_for_status()
        return ModelVersion.model_validate(r.json())


class Studio:
    """Top-level SDK entrypoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._http = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self.projects = ProjectsAPI(self._http)
        self.datasets = DatasetsAPI(self._http)
        self.jobs = JobsAPI(self._http)
        self.experiments = ExperimentsAPI(self._http)
        self.models = ModelsAPI(self._http)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Studio:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def system_info(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._http.get("/system/info").raise_for_status().json())

    def settings(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._http.get("/settings").raise_for_status().json())
