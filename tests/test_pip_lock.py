"""Exercise the lock workaround with a real index missing wheel hashes."""

import functools
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import pip
import pytest
import subprocess
import sys
import threading
import tomllib
import zipfile

from cellonaut.release_checks.dependency_lock import PIP_VERSION


def test_lock_downloads_unhashed_wheel_instead_of_only_metadata(tmp_path):
    if pip.__version__ != PIP_VERSION:
        pytest.skip(f"pip-lock workaround is specific to pip {PIP_VERSION}")
    metadata = b"Metadata-Version: 2.1\nName: lock-probe\nVersion: 1.0\n"
    index = tmp_path / "simple" / "lock-probe"
    index.mkdir(parents=True)
    wheel = index / "lock_probe-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("lock_probe-1.0.dist-info/METADATA", metadata)
        archive.writestr("lock_probe-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr("lock_probe-1.0.dist-info/RECORD", "")
    wheel.with_suffix(".whl.metadata").write_bytes(metadata)
    (index / "index.html").write_text(
        f'<a href="{wheel.name}" data-core-metadata="true">{wheel.name}</a>', encoding="utf-8"
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(SimpleHTTPRequestHandler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "pylock.toml"
    args = ["--isolated", "--no-cache-dir", "--disable-pip-version-check",
            "--index-url", f"http://127.0.0.1:{server.server_port}/simple",
            "--output", str(output), "lock-probe==1.0"]
    try:
        baseline = subprocess.run([sys.executable, "-m", "pip", "lock", *args], capture_output=True, text=True, timeout=30)
        assert baseline.returncode != 0
        assert "NotImplementedError" in baseline.stderr
        fixed = subprocess.run([sys.executable, "-m", "cellonaut.release_checks.pip_lock", *args], capture_output=True, text=True, timeout=30)
        assert fixed.returncode == 0, fixed.stdout + fixed.stderr
        package = tomllib.loads(output.read_text(encoding="utf-8"))["packages"][0]
        assert package["wheels"][0]["hashes"]["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
