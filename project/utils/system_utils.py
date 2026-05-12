from __future__ import annotations
import os
import psutil
import torch
import cv2


def detect_backends() -> dict:
    cuda_torch = torch.cuda.is_available()
    cuda_cv = False
    try:
        cuda_cv = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        cuda_cv = False
    return {
        "torch_cuda": cuda_torch,
        "opencv_cuda": cuda_cv,
        "device": "cuda" if cuda_torch else "cpu",
        "cpu_count": os.cpu_count() or 1,
    }


def system_metrics() -> dict:
    vm = psutil.virtual_memory()
    return {
        "ram_used_gb": round((vm.total - vm.available) / (1024**3), 2),
        "ram_total_gb": round(vm.total / (1024**3), 2),
    }
