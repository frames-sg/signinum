# CUDA Runner Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten the manual CUDA runner validation job so it proves CUDA adapter compatibility behavior without implying runtime CUDA decode.

**Architecture:** Keep the existing `.github/workflows/gpu-validation.yml` split between Metal and CUDA jobs. Add a repo-integrity regression test that extracts the CUDA job block and asserts it contains CUDA-focused diagnostics and adapter tests while rejecting the accidental J2K Metal bench command. Update only the manual workflow and the integrity test unless documentation verification shows a mismatch.

**Tech Stack:** GitHub Actions YAML, Rust integration tests, Cargo test/bench commands.

---

## File Structure

- Modify `.github/workflows/gpu-validation.yml`: add CUDA runner diagnostics and remove the Metal bench compile command from the CUDA job.
- Modify `crates/signinum-core/tests/repo_integrity.rs`: add a CUDA job extraction helper and a regression test for CUDA job focus.
- Read-only verification for `docs/wsi-decode-api.md`, `docs/release.md`, and `docs/bench.md`: confirm they still state compatibility-only CUDA validation and no runtime CUDA decode claim.

### Task 1: Add CUDA Workflow Contract Regression Test

**Files:**
- Modify: `crates/signinum-core/tests/repo_integrity.rs`

- [ ] **Step 1: Write the failing test**

Add this test after `gpu_validation_workflow_is_self_hosted_and_explicit`:

```rust
#[test]
fn cuda_gpu_validation_job_stays_cuda_focused() {
    let root = repo_root();
    let workflow_path = root.join(".github/workflows/gpu-validation.yml");
    let workflow = fs::read_to_string(&workflow_path).expect("read GPU validation workflow");
    let cuda_job = workflow_job(&workflow, "cuda-x86_64-compatibility");

    for required in [
        "runs-on: [self-hosted, Linux, X64, cuda]",
        "uname -a",
        "rustc -Vv",
        "cargo -V",
        "nvidia-smi",
        "CUDA adapter tests do not require runtime CUDA",
        "cargo test -p signinum-jpeg-cuda --all-targets --features cuda-runtime",
        "cargo test -p signinum-j2k-cuda --all-targets --features cuda-runtime",
        "cargo bench -p signinum-jpeg --no-run",
    ] {
        assert!(
            cuda_job.contains(required),
            "{} CUDA job must contain `{required}`",
            workflow_path
                .strip_prefix(root)
                .unwrap_or(&workflow_path)
                .display()
        );
    }

    for forbidden in [
        "cargo bench -p signinum-j2k-metal --bench compare --no-run",
        "cargo test -p signinum-jpeg-metal",
        "cargo test -p signinum-j2k-metal",
    ] {
        assert!(
            !cuda_job.contains(forbidden),
            "{} CUDA job must not contain Metal validation command `{forbidden}`",
            workflow_path
                .strip_prefix(root)
                .unwrap_or(&workflow_path)
                .display()
        );
    }
}
```

Add this helper before `rust_sources`:

```rust
fn workflow_job<'a>(workflow: &'a str, job_name: &str) -> &'a str {
    let marker = format!("  {job_name}:");
    let start = workflow
        .find(&marker)
        .unwrap_or_else(|| panic!("missing workflow job {job_name}"));
    let rest = &workflow[start..];
    let mut search_start = marker.len();
    let mut end = rest.len();
    while let Some(relative) = rest[search_start..].find("\n  ") {
        let candidate = search_start + relative + 1;
        if !rest[candidate..].starts_with("    ") {
            end = candidate;
            break;
        }
        search_start = candidate + 1;
    }
    &rest[..end]
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```sh
cargo test -p signinum-core cuda_gpu_validation_job_stays_cuda_focused -- --exact
```

Expected: FAIL because the CUDA job does not yet contain runner diagnostics and still contains:

```text
cargo bench -p signinum-j2k-metal --bench compare --no-run
```

### Task 2: Update CUDA GPU Validation Workflow

**Files:**
- Modify: `.github/workflows/gpu-validation.yml`

- [ ] **Step 1: Replace the CUDA job steps**

Update only the `cuda-x86_64-compatibility` job steps to this shape:

```yaml
  cuda-x86_64-compatibility:
    name: CUDA API compatibility on x86_64
    runs-on: [self-hosted, Linux, X64, cuda]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Show CUDA runner diagnostics
        run: |
          uname -a
          rustc -Vv
          cargo -V
          if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi
          else
            echo "nvidia-smi not found; CUDA adapter tests do not require runtime CUDA"
          fi
      - run: cargo test -p signinum-jpeg-cuda --all-targets --features cuda-runtime
      - run: cargo test -p signinum-j2k-cuda --all-targets --features cuda-runtime
      - run: cargo bench -p signinum-jpeg --no-run
```

The Metal job stays unchanged.

- [ ] **Step 2: Run test to verify it passes**

Run:

```sh
cargo test -p signinum-core cuda_gpu_validation_job_stays_cuda_focused -- --exact
```

Expected: PASS.

### Task 3: Verify Existing CUDA Compatibility Behavior

**Files:**
- No planned edits.

- [ ] **Step 1: Run JPEG CUDA adapter tests**

Run:

```sh
cargo test -p signinum-jpeg-cuda --all-targets --features cuda-runtime
```

Expected: PASS. The tests should still prove `Auto`/`Cpu` host surfaces and explicit CUDA unavailability.

- [ ] **Step 2: Run J2K CUDA adapter tests**

Run:

```sh
cargo test -p signinum-j2k-cuda --all-targets --features cuda-runtime
```

Expected: PASS. The tests should still prove `Auto`/`Cpu` host surfaces and explicit CUDA unavailability.

- [ ] **Step 3: Run touched integrity tests**

Run:

```sh
cargo test -p signinum-core gpu_validation
```

Expected: PASS for both workflow integrity tests.

### Task 4: Documentation Contract Check

**Files:**
- Read: `docs/wsi-decode-api.md`
- Read: `docs/release.md`
- Read: `docs/bench.md`

- [ ] **Step 1: Check docs retain compatibility-only CUDA language**

Run:

```sh
rg -n "compatibility-only|no runtime CUDA decode|not a runtime CUDA decode|do not establish runtime CUDA decode" docs/wsi-decode-api.md docs/release.md docs/bench.md
```

Expected: output includes all three docs and no contradictory runtime CUDA claim.

- [ ] **Step 2: Patch docs only if the check exposes a mismatch**

If a doc contradicts compatibility-only validation, replace the contradictory sentence with this wording:

```markdown
Passing the CUDA self-hosted job validates compatibility-only adapter behavior
and build compatibility; it is not a runtime CUDA decode or performance claim.
```

If the check shows docs are already aligned, make no documentation edit.

### Task 5: Final Verification and Commit

**Files:**
- Modify: `.github/workflows/gpu-validation.yml`
- Modify: `crates/signinum-core/tests/repo_integrity.rs`
- Optional modify only if needed: `docs/wsi-decode-api.md`, `docs/release.md`, `docs/bench.md`

- [ ] **Step 1: Run format check**

Run:

```sh
cargo fmt --check
```

Expected: PASS.

- [ ] **Step 2: Run narrow final test set**

Run:

```sh
cargo test -p signinum-core gpu_validation
cargo test -p signinum-jpeg-cuda --all-targets --features cuda-runtime
cargo test -p signinum-j2k-cuda --all-targets --features cuda-runtime
```

Expected: all commands PASS.

- [ ] **Step 3: Review diff**

Run:

```sh
git diff -- .github/workflows/gpu-validation.yml crates/signinum-core/tests/repo_integrity.rs docs/wsi-decode-api.md docs/release.md docs/bench.md
```

Expected: diff is limited to the CUDA runner validation workflow, repo-integrity test, and only necessary docs wording.

- [ ] **Step 4: Commit implementation**

Run:

```sh
git add .github/workflows/gpu-validation.yml crates/signinum-core/tests/repo_integrity.rs docs/wsi-decode-api.md docs/release.md docs/bench.md
git commit -m "ci: harden cuda runner validation"
```

Expected: commit succeeds and does not include unrelated worktree changes.

## Self-Review

- Spec coverage: workflow diagnostics, CUDA adapter tests, no runtime CUDA claim, no Metal command in CUDA job, repo-integrity coverage, and docs contract are all represented in tasks.
- Gap scan: no unresolved planning language remains.
- Type consistency: test names and helper names are consistent across write, run, and verification steps.
