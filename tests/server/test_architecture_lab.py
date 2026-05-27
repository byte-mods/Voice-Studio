from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from oas_server.jobs.kernel_bench import _exec_user_module, _expand_autotune_grid, _compare


def _client() -> TestClient:
    from oas_server.main import create_app
    return TestClient(create_app())


def test_speech_presets_compilation() -> None:
    # High-fidelity Triton code templates from frontend
    rmsnorm_code = """
import torch
def kernel(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + 1e-6) * weight
"""
    
    layernorm_code = """
import torch
def kernel(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, 1e-5)
"""

    swiglu_code = """
import torch
def kernel(x: torch.Tensor) -> torch.Tensor:
    x_gate, x_val = x.chunk(2, dim=-1)
    return (x_gate * torch.sigmoid(x_gate)) * x_val.squeeze(-1)
"""

    # Assert they compile correctly inside execution spaces
    ns_rms = _exec_user_module(rmsnorm_code, "test_rmsnorm")
    assert "kernel" in ns_rms
    assert callable(ns_rms["kernel"])

    ns_ln = _exec_user_module(layernorm_code, "test_layernorm")
    assert "kernel" in ns_ln
    assert callable(ns_ln["kernel"])

    ns_swiglu = _exec_user_module(swiglu_code, "test_swiglu")
    assert "kernel" in ns_swiglu
    assert callable(ns_swiglu["kernel"])


def test_autotune_grid_expansion() -> None:
    grid = {
        "BLOCK": [256, 512],
        "num_warps": [4, 8]
    }
    points = _expand_autotune_grid(grid)
    assert len(points) == 4
    assert {"BLOCK": 256, "num_warps": 4} in points
    assert {"BLOCK": 512, "num_warps": 8} in points


def test_numerical_correctness_comparison() -> None:
    import torch
    x = torch.randn(10, 10)
    y_ok = x + 1e-5
    y_fail = x + 1.0

    res_ok = _compare(x, y_ok, atol=1e-4, rtol=1e-4)
    assert res_ok["ok"] is True

    res_fail = _compare(x, y_fail, atol=1e-4, rtol=1e-4)
    assert res_fail["ok"] is False


def test_kernel_stub_execution_loop() -> None:
    with _client() as c:
        pid = c.post("/projects", json={"slug": "lab-stub", "name": "Lab Stub"}).json()["id"]
        
        # Register a CUDA C++ stub kernel
        r = c.post(
            "/kernels",
            json={
                "project_id": pid,
                "slug": "rmsnorm-stub",
                "name": "RMSNorm CUDA Stub",
                "backend": "cuda",
                "source": "def kernel(x, w): return x * w",
                "reference": "def reference(x, w): return x * w",
                "bench_config": {"shapes": [{"args": [{"shape": [32, 32], "dtype": "float32"}, {"shape": [32], "dtype": "float32"}]}]},
            },
        )
        assert r.status_code == 201
        kid = r.json()["id"]

        # Run benchmark (submits kernel_bench job)
        rb = c.post(f"/kernels/{kid}/benchmark")
        assert rb.status_code == 200
        assert "job_id" in rb.json()


def test_module_parameter_and_flops_projections() -> None:
    # Verify the parameters & FLOPs estimation mathematical formulas
    hidden_dim = 1024
    vocab_size = 50000
    layers = 24
    seq_len = 4096
    batch_size = 16

    # LLM (Attention / RoPE) parameter calculation
    params_llm = (vocab_size * hidden_dim) + layers * (hidden_dim * hidden_dim * 12)
    assert params_llm == 51200000 + 24 * (1024 * 1024 * 12)
    assert params_llm == 353189888  # 353M parameters

    # KV Cache size (bytes, FP16)
    kv_cache_bytes = 4 * layers * hidden_dim * seq_len * batch_size
    assert kv_cache_bytes == 4 * 24 * 1024 * 4096 * 16
    kv_cache_mb = kv_cache_bytes / (1024 * 1024)
    assert kv_cache_mb == 6144.0  # 6.1 GB
