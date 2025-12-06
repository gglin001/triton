import os
import torch
import torch.cpu

import triton
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver
from triton.backends import nvidia


TRITON_MOCK_PTX_VERSION = 87
TRITON_MOCK_WARP_SIZE = 32
TRITON_MOCK_ARCH = 90

os.environ["TRITON_ALWAYS_COMPILE"] = "1"
os.environ["TRITON_MOCK_PTX_VERSION"] = f"{TRITON_MOCK_PTX_VERSION}"


class FakeCUDABackend(nvidia.compiler.CUDABackend):
    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"


class FakeCudaDriver(GPUDriver):
    def __init__(self):
        super().__init__()
        import torch
        import torch.cpu

        self.get_current_device = torch.cpu.current_device
        self.set_current_device = torch.cpu.set_device
        self.get_current_stream = torch.cpu.current_stream

    def get_current_target(self):
        warp_size = 32
        capability = TRITON_MOCK_ARCH
        return GPUTarget("cpu", capability, warp_size)

    def get_active_torch_device(self):
        import torch

        return torch.device("cpu")

    def get_device_interface(self):
        import torch

        return torch.cpu

    @staticmethod
    def is_active():
        return True

    def map_python_to_cpp_type(self, ty: str) -> str:
        return nvidia.driver.ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device="cpu")

    def clear_cache(self, cache):
        cache.zero_()


nvidia.compiler.CUDABackend = FakeCUDABackend  # noqa
# not work
nvidia.compiler.supports_target = FakeCUDABackend.supports_target  # noqa
nvidia.driver.CudaDriver = FakeCudaDriver  # noqa

triton.runtime.driver.set_active(FakeCudaDriver())

device = "cpu"
torch.cpu.set_device(device)
