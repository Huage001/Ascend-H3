"""Small standard-library client for the pinned Omni synchronous video API."""

import json
import time
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from h3_ascend.paths import source_root


def request_fields(case, prompt, steps=8, seed=42):
    workloads = json.loads((source_root() / "configs/workloads.json").read_text())
    shape = workloads[case]
    if not prompt.strip():
        raise ValueError("The prompt must not be empty")
    if steps < 1:
        raise ValueError("steps must be positive")
    return {
        "prompt": prompt,
        "width": shape["width"],
        "height": shape["height"],
        "num_frames": shape["frames"],
        "fps": shape["fps"],
        "num_inference_steps": steps,
        "seed": seed,
        "enable_frame_interpolation": "false",
        "return_stage_metrics": "true",
    }


def multipart(fields):
    boundary = "ascend-h3-" + uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), "multipart/form-data; boundary=" + boundary


def generate(args):
    fields = request_fields(args.case, args.prompt_file.read_text(), args.steps, args.seed)
    body, content_type = multipart(fields)
    request = Request(
        args.url.rstrip("/") + "/v1/videos/sync",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=7200) as response:
            if response.headers.get_content_type() != "video/mp4":
                raise RuntimeError("Server returned a non-video response")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            partial = args.output.with_suffix(args.output.suffix + ".partial")
            with partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            partial.replace(args.output)
            metadata = dict(response.headers)
    except HTTPError as error:
        raise RuntimeError(
            f"Omni HTTP {error.code}: {error.read().decode(errors='replace')}"
        ) from error
    receipt = {
        "request": fields,
        "headers": metadata,
        "client_wall_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False))
    print(args.output.resolve())
