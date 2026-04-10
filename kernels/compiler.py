import os
import glob
import platform
import shutil
import subprocess
import torch
from torch.utils.cpp_extension import load_inline


def _setup_msvc():
    if shutil.which("cl"):
        return True
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if not os.path.isfile(vswhere):
        return False
    try:
        vs_path = subprocess.check_output(
            [vswhere, "-latest", "-property", "installationPath",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"],
            text=True
        ).strip()
        vcvarsall = os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvarsall.bat")
        if not os.path.isfile(vcvarsall):
            return False
        output = subprocess.check_output(
            f'cmd /c ""{vcvarsall}" x64 && set"',
            text=True, shell=True
        )
        for line in output.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ[key] = value
        return True
    except Exception:
        return False


def _is_kernel_cached(name):
    try:
        from torch.utils.cpp_extension import _get_build_directory
        build_dir = _get_build_directory(name, verbose=False)
        return os.path.isdir(build_dir) and any(f.endswith(('.so', '.pyd')) for f in os.listdir(build_dir))
    except Exception:
        return False


_compiled_kernels = {}


_COMMON_CUDA_HEADER = r"""
static __device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

static __device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float wrs[8];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warpReduceSum(val);
    if (lane == 0) wrs[wid] = val;
    __syncthreads();
    val = (threadIdx.x < 8) ? wrs[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

static __device__ __forceinline__ void warpReduceOnlineSoftmax(float& m, float& d) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float m2 = __shfl_down_sync(0xffffffff, m, offset);
        float d2 = __shfl_down_sync(0xffffffff, d, offset);
        float m_new = fmaxf(m, m2);
        d = d * __expf(m - m_new) + d2 * __expf(m2 - m_new);
        m = m_new;
    }
}

static __device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        if (__int_as_float(expected) >= val) return;
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (old != expected);
}

static __device__ __forceinline__ void blockReduceOnlineSoftmax(float* reduce_buf, float m, float d) {
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    warpReduceOnlineSoftmax(m, d);
    if (lane == 0) {
        reduce_buf[wid] = m;
        reduce_buf[wid + 8] = d;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float gm = reduce_buf[0]; float gd = reduce_buf[8];
        for (int w = 1; w < 8; w++) {
            float m2 = reduce_buf[w]; float d2 = reduce_buf[w + 8];
            float m_new = fmaxf(gm, m2);
            gd = gd * __expf(gm - m_new) + d2 * __expf(m2 - m_new);
            gm = m_new;
        }
        reduce_buf[0] = gm; reduce_buf[8] = gd;
    }
    __syncthreads();
}
"""


def _compile_kernel_with_header(name, cpp_source, cuda_source, functions):
    return _compile_kernel(name, cpp_source, _COMMON_CUDA_HEADER + cuda_source, functions)


def _get_platform_flags():
    if platform.system() == "Windows":
        return ["/permissive-"], ["-O3", "-Xcompiler", "/permissive-"]
    return [], ["-O3"]


def _compile_kernel(name, cpp_source, cuda_source, functions):
    if name in _compiled_kernels:
        return _compiled_kernels[name]

    needs_build = not _is_kernel_cached(name)
    if needs_build:
        print(f"Compiling CUDA kernel '{name}' (first run only)...")
    else:
        # Clean stale lock files from killed compilations to prevent deadlock
        # (PyTorch's FileBaton spins forever waiting for a lock file to be deleted)
        try:
            from torch.utils.cpp_extension import _get_build_directory
            lock_file = os.path.join(_get_build_directory(name, verbose=False), 'lock')
            if os.path.exists(lock_file):
                os.remove(lock_file)
        except Exception:
            pass

    extra_cflags, extra_cuda_cflags = _get_platform_flags()
    try:
        module = load_inline(
            name=name,
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=functions,
            verbose=False,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=extra_cflags,
        )
        _compiled_kernels[name] = module
        return module
    except Exception as e:
        import warnings
        msg = f"CUDA kernel '{name}' compilation failed, using PyTorch fallback.\n"
        msg += f"  Error: {e}\n"
        if platform.system() == "Windows":
            msg += "  Install Visual Studio Build Tools with 'Desktop development with C++' workload.\n"
            msg += "  Ensure CUDA Toolkit is installed matching your PyTorch CUDA version."
        else:
            msg += "  Ensure gcc/g++ and CUDA Toolkit are installed matching your PyTorch CUDA version."
        warnings.warn(msg)
        _compiled_kernels[name] = None
        return None


_cuda_available = False
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}")
    if platform.system() == "Windows":
        for scripts_dir in glob.glob(os.path.join(os.environ.get("APPDATA", ""), "Python", "Python*", "Scripts")):
            if os.path.isdir(scripts_dir) and scripts_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = scripts_dir + os.pathsep + os.environ["PATH"]
                break
        _setup_msvc()
    _cuda_available = True
