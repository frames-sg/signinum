# SPDX-License-Identifier: MIT OR Apache-2.0

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import urllib.error
import urllib.request
import unittest
import zipfile
from typing import Any, Mapping

from scripts import github_actions_verify as verifier


SHA = "a" * 40
STALE_SHA = "b" * 40
TAG_SHA = "c" * 40


class FakeApi:
    def __init__(self) -> None:
        self.responses: dict[tuple[str, tuple[tuple[str, str], ...]], Any] = {}
        self.downloads: dict[str, bytes] = {}
        self.calls: list[tuple[str, tuple[tuple[str, str], ...]]] = []

    @staticmethod
    def key(
        path: str, params: Mapping[str, str | int] | None = None
    ) -> tuple[str, tuple[tuple[str, str], ...]]:
        return path, tuple(sorted((key, str(value)) for key, value in (params or {}).items()))

    def add(
        self,
        path: str,
        payload: Any,
        params: Mapping[str, str | int] | None = None,
    ) -> None:
        self.responses[self.key(path, params)] = payload

    def get_json(
        self, path: str, params: Mapping[str, str | int] | None = None
    ) -> Any:
        key = self.key(path, params)
        self.calls.append(key)
        if key not in self.responses:
            raise AssertionError(f"unexpected fake API request: {key}")
        response = self.responses[key]
        if isinstance(response, Exception):
            raise response
        return response

    def get_optional_json(
        self, path: str, params: Mapping[str, str | int] | None = None
    ) -> verifier.OptionalJsonResponse:
        response = self.get_json(path, params)
        return verifier.OptionalJsonResponse(found=response is not None, payload=response)

    def add_download(self, path: str, payload: bytes) -> None:
        self.downloads[path] = payload

    def download_bytes(self, path: str, *, maximum_bytes: int) -> bytes:
        del maximum_bytes
        if path not in self.downloads:
            raise AssertionError(f"unexpected fake API download: {path}")
        return self.downloads[path]


def workflow_metadata(api: FakeApi, filename: str, workflow_id: int) -> None:
    api.add(
        f"/actions/workflows/{filename}",
        {"id": workflow_id, "path": f".github/workflows/{filename}"},
    )


def workflow_run(
    run_id: int,
    *,
    sha: str = SHA,
    workflow_id: int = 77,
    path: str = ".github/workflows/gpu-validation.yml",
    status: str = "completed",
    conclusion: str | None = "success",
    event: str = "workflow_dispatch",
    head_branch: str | None = "main",
) -> dict[str, Any]:
    return {
        "id": run_id,
        "head_sha": sha,
        "workflow_id": workflow_id,
        "path": path,
        "status": status,
        "conclusion": conclusion,
        "event": event,
        "head_branch": head_branch,
    }


def workflow_job(
    name: str, status: str = "completed", conclusion: str | None = "success"
) -> dict[str, Any]:
    return {"name": name, "status": status, "conclusion": conclusion}


def add_runs(api: FakeApi, workflow_id: int, runs: list[dict[str, Any]]) -> None:
    api.add(
        f"/actions/workflows/{workflow_id}/runs",
        {"workflow_runs": runs},
        {"head_sha": SHA, "per_page": 100, "page": 1},
    )


def add_jobs(api: FakeApi, run_id: int, jobs: list[dict[str, Any]]) -> None:
    api.add(
        f"/actions/runs/{run_id}/jobs",
        {"jobs": jobs},
        {"per_page": 100, "page": 1},
    )


def report_archive(stem: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{stem}.json", "{}\n")
        archive.writestr(f"{stem}.md", "# evidence\n")
    return output.getvalue()


class PullRequestPolicyTests(unittest.TestCase):
    def test_reusable_runner_changes_require_both_exact_nested_jobs(self) -> None:
        for path in (
            ".github/workflows/gpu-validation-runner.yml",
            ".github/workflows/gpu-benchmarks-runner.yml",
        ):
            with self.subTest(path=path):
                decision = verifier.classify_gpu_paths([path])
                self.assertEqual(
                    decision.required_jobs,
                    ("GPU / CUDA quick validation", "GPU / Metal quick validation"),
                )

    def test_pull_request_files_are_paginated_and_renames_use_both_paths(self) -> None:
        api = FakeApi()
        first_page = [{"filename": f"docs/generated-{index}.md"} for index in range(100)]
        api.add(
            "/pulls/42/files",
            first_page,
            {"per_page": 100, "page": 1},
        )
        api.add(
            "/pulls/42/files",
            [
                {
                    "filename": "archive/old.rs",
                    "previous_filename": "crates/j2k-cuda/src/old.rs",
                },
                {"filename": "scripts/github_actions_verify.py"},
            ],
            {"per_page": 100, "page": 2},
        )

        paths = verifier.fetch_pull_request_paths(api, 42)  # type: ignore[arg-type]
        decision = verifier.classify_gpu_paths(paths)

        self.assertEqual(
            decision.required_jobs,
            (verifier.CUDA_QUICK_JOB, verifier.METAL_QUICK_JOB),
        )
        self.assertEqual(
            decision.changed_gpu_paths,
            (
                "crates/j2k-cuda/src/old.rs",
                "scripts/github_actions_verify.py",
            ),
        )
        self.assertEqual(len(api.calls), 2)

    def test_non_gpu_paths_require_no_hardware_jobs(self) -> None:
        decision = verifier.classify_gpu_paths(["README.md", "crates/j2k/src/lib.rs"])
        self.assertEqual(decision.required_jobs, ())
        self.assertEqual(decision.changed_gpu_paths, ())

    def test_shared_conformance_and_routing_paths_require_both_hardware_lanes(self) -> None:
        paths = [
            "crates/j2k-t803/src/runner.rs",
            "corpus/j2k-conformance/t803-v3.toml",
            "xtask/src/t803.rs",
            "xtask/src/auto_routing/tests.rs",
        ]

        decision = verifier.classify_gpu_paths(paths)

        self.assertEqual(
            decision.required_jobs,
            (verifier.CUDA_QUICK_JOB, verifier.METAL_QUICK_JOB),
        )
        self.assertEqual(decision.changed_gpu_paths, tuple(sorted(paths)))


class ParserTests(unittest.TestCase):
    def test_verify_candidate_parser_smoke(self) -> None:
        args = verifier.build_parser().parse_args(
            [
                "verify-candidate",
                "--repository",
                "frames-sg/j2k",
                "--candidate-sha",
                SHA,
                "--t803-out-dir",
                "target/t803/release-evidence",
            ]
        )
        self.assertEqual(args.command, "verify-candidate")
        self.assertEqual(args.aggregate_job, verifier.RELEASE_CANDIDATE_JOB)
        self.assertEqual(args.cuda_job, verifier.CUDA_JOB)
        self.assertEqual(args.metal_job, verifier.METAL_JOB)
        self.assertEqual(args.t803_out_dir, "target/t803/release-evidence")


class RedirectSecurityTests(unittest.TestCase):
    def test_cross_origin_redirect_strips_every_credential_header(self) -> None:
        request = urllib.request.Request(
            "https://api.github.com/repos/frames-sg/j2k/actions/artifacts/1/zip",
            headers={
                "Authorization": "Bearer secret",
                "Cookie": "session=secret",
                "Proxy-Authorization": "Basic secret",
                "X-Trace": "keep-me",
            },
        )

        redirected = verifier._CredentialSafeRedirectHandler().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://objects.githubusercontent.com/artifact.zip",
        )

        self.assertIsNotNone(redirected)
        assert redirected is not None
        headers = {name.lower(): value for name, value in redirected.header_items()}
        self.assertNotIn("authorization", headers)
        self.assertNotIn("cookie", headers)
        self.assertNotIn("proxy-authorization", headers)
        self.assertEqual(headers["x-trace"], "keep-me")

    def test_verify_release_parser_requires_origin_context(self) -> None:
        args = verifier.build_parser().parse_args(
            [
                "verify-release",
                "--repository",
                "frames-sg/j2k",
                "--origin-url",
                "https://github.com/frames-sg/j2k.git",
                "--server-url",
                "https://github.com",
                "--tag",
                "v0.7.0",
                "--candidate-sha",
                SHA,
                "--t803-out-dir",
                "target/t803/release-evidence",
            ]
        )
        self.assertEqual(args.command, "verify-release")
        self.assertEqual(args.origin_url, "https://github.com/frames-sg/j2k.git")
        self.assertEqual(args.t803_out_dir, "target/t803/release-evidence")


class WorkflowVerificationTests(unittest.TestCase):
    def test_runs_and_jobs_are_paginated_and_one_run_contains_every_job(self) -> None:
        api = FakeApi()
        workflow_metadata(api, "gpu-validation.yml", 77)
        stale_runs = [workflow_run(index + 1, sha=STALE_SHA) for index in range(100)]
        api.add(
            "/actions/workflows/77/runs",
            {"workflow_runs": stale_runs},
            {"head_sha": SHA, "per_page": 100, "page": 1},
        )
        api.add(
            "/actions/workflows/77/runs",
            {"workflow_runs": [workflow_run(500)]},
            {"head_sha": SHA, "per_page": 100, "page": 2},
        )
        filler_jobs = [workflow_job(f"filler {index}") for index in range(100)]
        api.add(
            "/actions/runs/500/jobs",
            {"jobs": filler_jobs},
            {"per_page": 100, "page": 1},
        )
        api.add(
            "/actions/runs/500/jobs",
            {"jobs": [workflow_job(verifier.CUDA_JOB), workflow_job(verifier.METAL_JOB)]},
            {"per_page": 100, "page": 2},
        )

        run_id = verifier.verify_workflow_run(
            api,  # type: ignore[arg-type]
            "gpu-validation.yml",
            SHA,
            [verifier.CUDA_JOB, verifier.METAL_JOB],
        )

        self.assertEqual(run_id, 500)
        self.assertIn(
            FakeApi.key(
                "/actions/workflows/77/runs",
                {"head_sha": SHA, "per_page": 100, "page": 2},
            ),
            api.calls,
        )
        self.assertIn(
            FakeApi.key(
                "/actions/runs/500/jobs", {"per_page": 100, "page": 2}
            ),
            api.calls,
        )

    def test_successes_from_different_runs_cannot_be_combined(self) -> None:
        api = FakeApi()
        workflow_metadata(api, "gpu-validation.yml", 77)
        add_runs(api, 77, [workflow_run(1), workflow_run(2)])
        add_jobs(api, 1, [workflow_job(verifier.CUDA_JOB)])
        add_jobs(api, 2, [workflow_job(verifier.METAL_JOB)])

        with self.assertRaisesRegex(verifier.VerificationError, "no acceptable single run"):
            verifier.verify_workflow_run(
                api,  # type: ignore[arg-type]
                "gpu-validation.yml",
                SHA,
                [verifier.CUDA_JOB, verifier.METAL_JOB],
            )

    def test_incomplete_skipped_missing_and_stale_evidence_is_rejected(self) -> None:
        cases = (
            (
                "in progress",
                workflow_run(1, status="in_progress", conclusion=None),
                [workflow_job(verifier.CUDA_JOB)],
            ),
            (
                "cancelled",
                workflow_run(1, conclusion="cancelled"),
                [workflow_job(verifier.CUDA_JOB)],
            ),
            (
                "skipped",
                workflow_run(1),
                [workflow_job(verifier.CUDA_JOB, conclusion="skipped")],
            ),
            (
                "failed",
                workflow_run(1),
                [workflow_job(verifier.CUDA_JOB, conclusion="failure")],
            ),
            ("missing", workflow_run(1), [workflow_job("not the required job")]),
            (
                "stale",
                workflow_run(1, sha=STALE_SHA),
                [workflow_job(verifier.CUDA_JOB)],
            ),
        )
        for label, run, jobs in cases:
            with self.subTest(label=label):
                api = FakeApi()
                workflow_metadata(api, "gpu-validation.yml", 77)
                add_runs(api, 77, [run])
                if run["head_sha"] == SHA and run["status"] == "completed":
                    add_jobs(api, 1, jobs)
                with self.assertRaises(verifier.VerificationError):
                    verifier.verify_workflow_run(
                        api,  # type: ignore[arg-type]
                        "gpu-validation.yml",
                        SHA,
                        [verifier.CUDA_JOB],
                    )

    def test_duplicate_required_job_names_are_rejected(self) -> None:
        api = FakeApi()
        workflow_metadata(api, "gpu-validation.yml", 77)
        add_runs(api, 77, [workflow_run(1)])
        add_jobs(
            api,
            1,
            [workflow_job(verifier.CUDA_JOB), workflow_job(verifier.CUDA_JOB)],
        )
        with self.assertRaisesRegex(verifier.VerificationError, "duplicate required jobs"):
            verifier.verify_workflow_run(
                api,  # type: ignore[arg-type]
                "gpu-validation.yml",
                SHA,
                [verifier.CUDA_JOB],
            )

    def test_required_event_and_branch_are_exact(self) -> None:
        api = FakeApi()
        workflow_metadata(api, "gpu-validation.yml", 77)
        add_runs(api, 77, [workflow_run(1, event="push", head_branch="feature")])
        with self.assertRaisesRegex(verifier.VerificationError, "expected workflow_dispatch"):
            verifier.verify_workflow_run(
                api,  # type: ignore[arg-type]
                "gpu-validation.yml",
                SHA,
                [verifier.CUDA_JOB],
                required_event="workflow_dispatch",
                required_head_branch="main",
            )

    def test_malformed_runs_payload_fails_closed(self) -> None:
        api = FakeApi()
        workflow_metadata(api, "gpu-validation.yml", 77)
        api.add(
            "/actions/workflows/77/runs",
            {"workflow_runs": "not-an-array"},
            {"head_sha": SHA, "per_page": 100, "page": 1},
        )
        with self.assertRaisesRegex(verifier.VerificationError, "must be an array"):
            verifier.verify_workflow_run(
                api,  # type: ignore[arg-type]
                "gpu-validation.yml",
                SHA,
                [verifier.CUDA_JOB],
            )

    def test_exact_workflow_path_is_required(self) -> None:
        api = FakeApi()
        api.add(
            "/actions/workflows/gpu-validation.yml",
            {"id": 77, "path": ".github/workflows/not-gpu-validation.yml"},
        )
        with self.assertRaisesRegex(verifier.VerificationError, "workflow path mismatch"):
            verifier.fetch_workflow_identity(  # type: ignore[arg-type]
                api, "gpu-validation.yml"
            )

    def test_exact_numeric_workflow_id_is_supported(self) -> None:
        api = FakeApi()
        api.add(
            "/actions/workflows/77",
            {"id": 77, "path": ".github/workflows/gpu-validation.yml"},
        )
        identity = verifier.fetch_workflow_identity(api, "77")  # type: ignore[arg-type]
        self.assertEqual(identity.workflow_id, 77)
        self.assertEqual(identity.path, ".github/workflows/gpu-validation.yml")


class ReleaseVerificationTests(unittest.TestCase):
    def test_cpu_t803_scope_does_not_require_a_gpu_run(self) -> None:
        specs = verifier.t803_artifact_specs(SHA, 10, None, scope="cpu")

        self.assertEqual(len(specs), 3)
        self.assertTrue(all(spec.run_id == 10 for spec in specs))
        self.assertTrue(all(spec.report_stem == "cpu" for spec in specs))

    def test_exact_run_t803_artifacts_are_downloaded_and_extracted(self) -> None:
        api = FakeApi()
        ci_run = 10
        gpu_run = 20
        specs = verifier.t803_artifact_specs(SHA, ci_run, gpu_run)
        for run_id in (ci_run, gpu_run):
            run_specs = [spec for spec in specs if spec.run_id == run_id]
            api.add(
                f"/actions/runs/{run_id}/artifacts",
                {
                    "artifacts": [
                        {
                            "id": index + run_id * 10,
                            "name": spec.artifact_name,
                            "expired": False,
                            "size_in_bytes": 512,
                            "workflow_run": {"id": run_id},
                        }
                        for index, spec in enumerate(run_specs)
                    ]
                },
                {"per_page": 100, "page": 1},
            )
            for index, spec in enumerate(run_specs):
                artifact_id = index + run_id * 10
                api.add_download(
                    f"/actions/artifacts/{artifact_id}/zip",
                    report_archive(spec.report_stem),
                )

        with tempfile.TemporaryDirectory() as directory:
            reports = verifier.download_t803_report_artifacts(
                api,  # type: ignore[arg-type]
                candidate_sha=SHA,
                ci_run_id=ci_run,
                gpu_run_id=gpu_run,
                output_dir=Path(directory),
            )

            self.assertEqual(len(reports), 5)
            self.assertTrue(all(path.is_file() for path in reports))
            self.assertEqual(
                {path.name for path in reports},
                {"cpu.json", "cuda.json", "metal.json"},
            )

    def test_t803_artifact_extraction_rejects_path_traversal(self) -> None:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("../cpu.json", "{}\n")
            archive.writestr("cpu.md", "# evidence\n")

        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            verifier.VerificationError, "unsafe"
        ):
            verifier.extract_t803_report_archive(
                output.getvalue(),
                report_stem="cpu",
                output_dir=Path(directory),
            )

    def test_post_freeze_candidate_verifies_ci_and_gpu_without_a_tag(self) -> None:
        api = FakeApi()
        api.add("/private-vulnerability-reporting", {"enabled": True})
        workflow_metadata(api, "ci.yml", 88)
        api.add(
            "/actions/workflows/88/runs",
            {
                "workflow_runs": [
                    workflow_run(
                        10,
                        workflow_id=88,
                        path=".github/workflows/ci.yml",
                        event="push",
                    )
                ]
            },
            {"head_sha": SHA, "per_page": 100, "page": 1},
        )
        add_jobs(api, 10, [workflow_job(verifier.RELEASE_CANDIDATE_JOB)])
        workflow_metadata(api, "gpu-validation.yml", 77)
        add_runs(api, 77, [workflow_run(20)])
        add_jobs(
            api,
            20,
            [workflow_job(verifier.CUDA_JOB), workflow_job(verifier.METAL_JOB)],
        )

        self.assertEqual(
            verifier.verify_candidate_evidence(
                api,  # type: ignore[arg-type]
                candidate_sha=SHA,
                ci_workflow="ci.yml",
                aggregate_job=verifier.RELEASE_CANDIDATE_JOB,
                gpu_workflow="gpu-validation.yml",
                cuda_job=verifier.CUDA_JOB,
                metal_job=verifier.METAL_JOB,
                ci_branch="main",
            ),
            (10, 20),
        )
        self.assertFalse(
            any(path.startswith("/git/") for path, _params in api.calls),
            "post-freeze candidate status must not require a release tag",
        )

    def test_cpu_candidate_scope_does_not_query_the_gpu_workflow(self) -> None:
        api = FakeApi()
        api.add("/private-vulnerability-reporting", {"enabled": True})
        workflow_metadata(api, "ci.yml", 88)
        api.add(
            "/actions/workflows/88/runs",
            {
                "workflow_runs": [
                    workflow_run(
                        10,
                        workflow_id=88,
                        path=".github/workflows/ci.yml",
                        event="push",
                    )
                ]
            },
            {"head_sha": SHA, "per_page": 100, "page": 1},
        )
        add_jobs(api, 10, [workflow_job(verifier.RELEASE_CANDIDATE_JOB)])

        result = verifier.verify_candidate_evidence(
            api,  # type: ignore[arg-type]
            candidate_sha=SHA,
            ci_workflow="ci.yml",
            aggregate_job=verifier.RELEASE_CANDIDATE_JOB,
            gpu_workflow="gpu-validation.yml",
            cuda_job=verifier.CUDA_JOB,
            metal_job=verifier.METAL_JOB,
            ci_branch="main",
            t803_scope="cpu",
        )

        self.assertEqual(result, (10, None))
        self.assertFalse(
            any("gpu-validation.yml" in path for path, _params in api.calls),
            "CPU evidence must remain independent of accelerator availability",
        )

    def test_cuda_candidate_scope_requires_only_the_cuda_job(self) -> None:
        api = FakeApi()
        api.add("/private-vulnerability-reporting", {"enabled": True})
        workflow_metadata(api, "ci.yml", 88)
        api.add(
            "/actions/workflows/88/runs",
            {
                "workflow_runs": [
                    workflow_run(
                        10,
                        workflow_id=88,
                        path=".github/workflows/ci.yml",
                        event="push",
                    )
                ]
            },
            {"head_sha": SHA, "per_page": 100, "page": 1},
        )
        add_jobs(api, 10, [workflow_job(verifier.RELEASE_CANDIDATE_JOB)])
        workflow_metadata(api, "gpu-validation.yml", 77)
        add_runs(api, 77, [workflow_run(20)])
        add_jobs(api, 20, [workflow_job(verifier.CUDA_JOB)])

        result = verifier.verify_candidate_evidence(
            api,  # type: ignore[arg-type]
            candidate_sha=SHA,
            ci_workflow="ci.yml",
            aggregate_job=verifier.RELEASE_CANDIDATE_JOB,
            gpu_workflow="gpu-validation.yml",
            cuda_job=verifier.CUDA_JOB,
            metal_job=verifier.METAL_JOB,
            ci_branch="main",
            t803_scope="cuda",
        )

        self.assertEqual(result, (10, 20))
        specs = verifier.t803_artifact_specs(SHA, 10, 20, scope="cuda")
        self.assertEqual(
            [(spec.run_id, spec.report_stem) for spec in specs], [(20, "cuda")]
        )

    def test_post_freeze_candidate_requires_private_vulnerability_reporting(self) -> None:
        cases = (
            ({"enabled": False}, "not enabled"),
            ({"enabled": "yes"}, "must be a boolean"),
            (
                verifier.VerificationError("GitHub API request failed with HTTP 403"),
                "HTTP 403",
            ),
        )
        for response, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                api = FakeApi()
                api.add("/private-vulnerability-reporting", response)
                with self.assertRaisesRegex(
                    verifier.VerificationError, expected_error
                ):
                    verifier.verify_candidate_evidence(
                        api,  # type: ignore[arg-type]
                        candidate_sha=SHA,
                        ci_workflow="ci.yml",
                        aggregate_job=verifier.RELEASE_CANDIDATE_JOB,
                        gpu_workflow="gpu-validation.yml",
                        cuda_job=verifier.CUDA_JOB,
                        metal_job=verifier.METAL_JOB,
                        ci_branch="main",
                    )
                self.assertEqual(
                    api.calls,
                    [("/private-vulnerability-reporting", ())],
                    "candidate verification must stop before accepting CI/GPU evidence",
                )

    def test_annotated_tag_is_peeled_and_ci_and_gpu_runs_are_exact(self) -> None:
        api = FakeApi()
        api.add("/private-vulnerability-reporting", {"enabled": True})
        api.add("/releases/tags/v0.7.0", None)
        api.add(
            "/git/ref/tags/v0.7.0",
            {
                "ref": "refs/tags/v0.7.0",
                "object": {"type": "tag", "sha": TAG_SHA},
            },
        )
        api.add(
            f"/git/tags/{TAG_SHA}",
            {"sha": TAG_SHA, "object": {"type": "commit", "sha": SHA}},
        )
        workflow_metadata(api, "ci.yml", 88)
        api.add(
            "/actions/workflows/88/runs",
            {
                "workflow_runs": [
                    workflow_run(
                        10,
                        workflow_id=88,
                        path=".github/workflows/ci.yml",
                        event="push",
                    )
                ]
            },
            {"head_sha": SHA, "per_page": 100, "page": 1},
        )
        add_jobs(api, 10, [workflow_job(verifier.RELEASE_CANDIDATE_JOB)])
        workflow_metadata(api, "gpu-validation.yml", 77)
        add_runs(api, 77, [workflow_run(20)])
        add_jobs(
            api,
            20,
            [workflow_job(verifier.CUDA_JOB), workflow_job(verifier.METAL_JOB)],
        )

        self.assertEqual(
            verifier.verify_release_evidence(
                api,  # type: ignore[arg-type]
                repository="frames-sg/j2k",
                origin_url="https://github.com/frames-sg/j2k.git",
                server_url="https://github.com",
                tag="v0.7.0",
                candidate_sha=SHA,
                ci_workflow="ci.yml",
                aggregate_job=verifier.RELEASE_CANDIDATE_JOB,
                gpu_workflow="gpu-validation.yml",
                cuda_job=verifier.CUDA_JOB,
                metal_job=verifier.METAL_JOB,
                ci_branch="main",
            ),
            (10, 20),
        )

    def test_lightweight_tag_and_candidate_mismatch_are_rejected(self) -> None:
        lightweight = FakeApi()
        lightweight.add(
            "/git/ref/tags/v0.7.0",
            {
                "ref": "refs/tags/v0.7.0",
                "object": {"type": "commit", "sha": SHA},
            },
        )
        with self.assertRaisesRegex(verifier.VerificationError, "must be annotated"):
            verifier.peel_annotated_tag(  # type: ignore[arg-type]
                lightweight, "v0.7.0"
            )

        annotated = FakeApi()
        annotated.add("/private-vulnerability-reporting", {"enabled": True})
        annotated.add("/releases/tags/v0.7.0", None)
        annotated.add(
            "/git/ref/tags/v0.7.0",
            {
                "ref": "refs/tags/v0.7.0",
                "object": {"type": "tag", "sha": TAG_SHA},
            },
        )
        annotated.add(
            f"/git/tags/{TAG_SHA}",
            {"sha": TAG_SHA, "object": {"type": "commit", "sha": STALE_SHA}},
        )
        with self.assertRaisesRegex(verifier.VerificationError, "not candidate"):
            verifier.verify_release_evidence(
                annotated,  # type: ignore[arg-type]
                repository="frames-sg/j2k",
                origin_url="https://github.com/frames-sg/j2k",
                server_url="https://github.com",
                tag="v0.7.0",
                candidate_sha=SHA,
                ci_workflow="ci.yml",
                aggregate_job=verifier.RELEASE_CANDIDATE_JOB,
                gpu_workflow="gpu-validation.yml",
                cuda_job=verifier.CUDA_JOB,
                metal_job=verifier.METAL_JOB,
                ci_branch="main",
            )

    def test_repository_origin_is_exact_and_credential_free(self) -> None:
        verifier.verify_repository_origin(
            "https://github.com/frames-sg/j2k.git",
            "https://github.com",
            "frames-sg/j2k",
        )
        for origin in (
            "git@github.com:frames-sg/j2k.git",
            "https://github.example/frames-sg/j2k.git",
            "https://github.com/frames-sg/j2k-other.git",
            "https://token@github.com/frames-sg/j2k.git",
            "https://github.com/frames-sg/j2k.git?mirror=true",
        ):
            with self.subTest(origin=origin), self.assertRaisesRegex(
                verifier.VerificationError, "origin does not match"
            ):
                verifier.verify_repository_origin(
                    origin, "https://github.com", "frames-sg/j2k"
                )
        with self.assertRaisesRegex(verifier.VerificationError, "server URL"):
            verifier.verify_repository_origin(
                "https://github.com/attacker/frames-sg/j2k",
                "https://github.com/attacker",
                "frames-sg/j2k",
            )

    def test_private_vulnerability_reporting_must_be_enabled(self) -> None:
        enabled = FakeApi()
        enabled.add("/private-vulnerability-reporting", {"enabled": True})
        verifier.require_private_vulnerability_reporting(enabled)  # type: ignore[arg-type]

        disabled = FakeApi()
        disabled.add("/private-vulnerability-reporting", {"enabled": False})
        with self.assertRaisesRegex(verifier.VerificationError, "not enabled"):
            verifier.require_private_vulnerability_reporting(  # type: ignore[arg-type]
                disabled
            )

        malformed = FakeApi()
        malformed.add("/private-vulnerability-reporting", {"enabled": "yes"})
        with self.assertRaisesRegex(verifier.VerificationError, "must be a boolean"):
            verifier.require_private_vulnerability_reporting(  # type: ignore[arg-type]
                malformed
            )

    def test_existing_github_release_in_any_state_is_rejected(self) -> None:
        for draft, prerelease, expected_state in (
            (True, False, "draft"),
            (False, True, "prerelease"),
            (False, False, "published"),
        ):
            with self.subTest(expected_state=expected_state):
                api = FakeApi()
                api.add(
                    "/releases/tags/v0.7.0",
                    {
                        "tag_name": "v0.7.0",
                        "draft": draft,
                        "prerelease": prerelease,
                    },
                )
                with self.assertRaisesRegex(
                    verifier.VerificationError, expected_state
                ):
                    verifier.require_github_release_absent(  # type: ignore[arg-type]
                        api, "v0.7.0"
                    )


class ApiFailureTests(unittest.TestCase):
    def test_missing_token_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            verifier.VerificationError, "GitHub API token is not configured"
        ):
            verifier.GitHubApi("https://api.github.invalid", "owner/repo", "")

    def test_artifact_download_uses_github_json_media_type(self) -> None:
        captured_headers: dict[str, str] = {}

        def capture(request: Any, timeout: int) -> Any:
            self.assertEqual(timeout, 30)
            captured_headers.update(
                {name.lower(): value for name, value in request.header_items()}
            )
            raise urllib.error.HTTPError(request.full_url, 418, "Stop", {}, None)

        api = verifier.GitHubApi(
            "https://api.github.invalid", "owner/repo", "token", opener=capture
        )
        with self.assertRaises(verifier.VerificationError):
            api.download_bytes("/actions/artifacts/1/zip", maximum_bytes=1024)

        self.assertEqual(captured_headers["accept"], "application/vnd.github+json")

    def test_http_failure_does_not_expose_token(self) -> None:
        token = "secret-token-that-must-not-appear"

        def reject(request: Any, timeout: int) -> Any:
            del timeout
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, None)

        api = verifier.GitHubApi(
            "https://api.github.invalid", "owner/repo", token, opener=reject
        )
        with self.assertRaises(verifier.VerificationError) as captured:
            api.get_json("/actions/workflows/ci.yml")
        self.assertNotIn(token, str(captured.exception))

    def test_only_http_404_is_optional_absence(self) -> None:
        def missing(request: Any, timeout: int) -> Any:
            del timeout
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

        api = verifier.GitHubApi(
            "https://api.github.invalid", "owner/repo", "token", opener=missing
        )
        self.assertFalse(api.get_optional_json("/releases/tags/v0.7.0").found)

        def forbidden(request: Any, timeout: int) -> Any:
            del timeout
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

        api = verifier.GitHubApi(
            "https://api.github.invalid", "owner/repo", "token", opener=forbidden
        )
        with self.assertRaisesRegex(verifier.VerificationError, "HTTP 403"):
            api.get_optional_json("/releases/tags/v0.7.0")


if __name__ == "__main__":
    unittest.main()
