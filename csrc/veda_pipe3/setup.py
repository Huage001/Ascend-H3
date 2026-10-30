import multiprocessing
import os
import shutil
import subprocess
from pathlib import Path

import torch
import torch_npu
from setuptools import find_packages, setup
from setuptools.command.build_clib import build_clib
from setuptools.command.build_ext import build_ext
from torch_npu.utils.cpp_extension import NpuExtension


ROOT = Path(__file__).resolve().parent
VERSION = "0.1.0"


class CMakeBuild(build_clib):
    def run(self):
        build_py = self.get_finalized_command("build_py")
        package_dir = Path(build_py.build_lib) / build_py.get_package_dir(
            "miowtion128pipe3_attention"
        )
        build_dir = ROOT / "build"
        native_lib_dir = build_dir / "lib"
        native_lib_dir.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(build_dir),
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={native_lib_dir}",
            f"-DTORCH_PATH={Path(torch.__file__).resolve().parent}",
            f"-DTORCH_NPU_PATH={Path(torch_npu.__file__).resolve().parent}",
            "-DCMAKE_BUILD_TYPE=RELEASE",
            f"-DGLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
        ]
        subprocess.check_call(cmake_args)

        jobs = os.getenv("MAX_JOBS", str(min(multiprocessing.cpu_count(), 8)))
        subprocess.check_call(
            ["cmake", "--build", str(build_dir), "--parallel", jobs]
        )

        destination = package_dir / "lib"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            native_lib_dir / "libmiowtion128pipe3_attention.so",
            destination / "libmiowtion128pipe3_attention.so",
        )


class BuildExt(build_ext):
    def run(self):
        self.run_command("build_clib")
        super().run()
        if self.inplace:
            # The Python package loads the native library directly from
            # miowtion128pipe3_attention/lib.  Keep an in-place build and the
            # source package in sync so local tests cannot silently exercise
            # a stale wheel artifact while the freshly built library sits
            # only below build/lib.*.
            source_lib_dir = ROOT / "miowtion128pipe3_attention" / "lib"
            source_lib_dir.mkdir(parents=True, exist_ok=True)
            destination = source_lib_dir / "libmiowtion128pipe3_attention.so"
            temporary = destination.with_suffix(".so.tmp")
            shutil.copy2(
                ROOT / "build" / "lib" / "libmiowtion128pipe3_attention.so",
                temporary,
            )
            temporary.replace(destination)


setup(
    name="ascend-block-sparse-attention",
    version=VERSION,
    description="Standalone Ascend BlockSparseAttention forward/backward extension",
    packages=find_packages(exclude=("tests", "tests.*")),
    package_data={"miowtion128pipe3_attention": ["lib/*.so"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=["torch", "torch-npu"],
    ext_modules=[NpuExtension("miowtion128pipe3_attention._C", sources=[])],
    cmdclass={"build_clib": CMakeBuild, "build_ext": BuildExt},
)
