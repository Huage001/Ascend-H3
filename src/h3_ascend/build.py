"""Compile source-only CANN operators into an external ABI-specific cache."""

import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from h3_ascend.native import LIBRARIES
from h3_ascend.paths import cache_root, runtime_root, source_root

SPARK = [
    "ascendc_inverse",
    "ascendc_meta32",
    "ascendc_partition_fast",
    "ascendc_permute_tail",
    "ascendc_preproc",
    "ascendc_proxy_four",
    "ascendc_proxy_onchip_v8",
    "ascendc_route_vcount",
    "ascendc_spark_joint_v4",
]
SPARK_LIB = {n: "libspark_" + n.removeprefix("ascendc_") + ".so" for n in SPARK}
SPARK_LIB.update(
    ascendc_proxy_onchip_v8="libspark_proxy_onchip.so",
    ascendc_spark_joint_v4="libspark_joint_v4.so",
)
ALIASES = {
    "ascendc_preproc": ["build", "build-v11"],
    "ascendc_partition_fast": ["build", "build-v2"],
    "ascendc_proxy_four": ["build", "build-v2"],
    "ascendc_proxy_onchip_v8": ["build", "build-v2"],
}


def build(method="all", soc="Ascend910B3", workers=2, jobs=4):
    import torch
    import torch_npu

    cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_CANN_PACKAGE_PATH")
    if not cann:
        raise RuntimeError("Source CANN set_env.sh before compiling operators")
    meta = dict(
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        abi=int(torch._C._GLIBCXX_USE_CXX11_ABI),
        cann=str(Path(cann).resolve()),
        soc=soc,
    )
    src = source_root() / "csrc"
    names = []
    if method in ("all", "spark"):
        names += SPARK
    if method in ("all", "vsa"):
        names += ["vsa_paired64"]
    if method in ("all", "veda"):
        names += [n for n in LIBRARIES if n.startswith("veda_")]
    if not names:
        return

    def one(name):
        p = src / name
        digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode())
        for f in sorted(p.rglob("*")):
            if f.is_file():
                digest.update(str(f.relative_to(p)).encode())
                digest.update(f.read_bytes())
        b = cache_root() / "build" / name / digest.hexdigest()[:20]
        b.mkdir(parents=True, exist_ok=True)
        lib = SPARK_LIB[name] if name in SPARK else LIBRARIES[name]
        log = b / "build.log"
        with log.open("w") as out:
            cmd = [
                sys.executable,
                "-m",
                "cmake",
                "-S",
                str(p),
                "-B",
                str(b),
                f"-DASCEND_CANN_PACKAGE_PATH={cann}",
                f"-DTORCH_PATH={Path(torch.__file__).parent}",
                f"-DTORCH_NPU_PATH={Path(torch_npu.__file__).parent}",
                f"-DGLIBCXX_USE_CXX11_ABI={meta['abi']}",
                f"-DSOC_VERSION={soc}",
            ]
            subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, check=True)
            subprocess.run(
                [sys.executable, "-m", "cmake", "--build", str(b), f"-j{jobs}"],
                stdout=out,
                stderr=subprocess.STDOUT,
                check=True,
            )
        built = b / lib
        if not built.is_file():
            raise FileNotFoundError(f"{built}; inspect {log}")
        if name in SPARK:
            for alias in ALIASES.get(name, ["build"]):
                target = runtime_root("spark") / "ops" / name / alias
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    target.unlink()
                target.symlink_to(b, target_is_directory=True)
        else:
            target = cache_root() / "lib" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                target.unlink()
            target.symlink_to(b, target_is_directory=True)
        (b / "build-info.json").write_text(
            json.dumps(
                dict(
                    meta,
                    source_digest=digest.hexdigest(),
                    library=lib,
                    library_sha256=hashlib.sha256(built.read_bytes()).hexdigest(),
                ),
                indent=2,
            )
        )
        print("BUILT", name, str(built), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, names))
