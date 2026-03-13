#!/usr/bin/env python3
"""Test file sharing services accessibility from current network."""

import subprocess
import tempfile
import time
from pathlib import Path


SERVICES = [
    {
        "name": "tmpfiles.org",
        "expiry": "1 hour",
        "limit": "100MB",
        "cmd": 'curl -s --connect-timeout 10 -F "file=@{file}" https://tmpfiles.org/api/v1/upload',
    },
    {
        "name": "uguu.se",
        "expiry": "48 hours",
        "limit": "128MB",
        "cmd": 'curl -s --connect-timeout 10 -F "files[]=@{file}" https://uguu.se/upload.php',
    },
    {
        "name": "x0.at",
        "expiry": "24 hours",
        "limit": "50MB",
        "cmd": 'curl -s --connect-timeout 10 -F "file=@{file}" https://x0.at/',
    },
    {
        "name": "litterbox",
        "expiry": "24 hours",
        "limit": "1GB",
        "cmd": 'curl -s --connect-timeout 10 -F "reqtype=fileupload" -F "time=24h" -F "fileToUpload=@{file}" https://litterbox.catbox.moe/resources/internals/api.php',
    },
    {
        "name": "catbox.moe",
        "expiry": "permanent",
        "limit": "200MB",
        "cmd": 'curl -s --connect-timeout 10 -F "reqtype=fileupload" -F "fileToUpload=@{file}" https://catbox.moe/user/api.php',
    },
]


def test_service(service: dict, test_file: str) -> dict:
    """Test a single service."""
    cmd = service["cmd"].format(file=test_file)
    start = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=15
        )
        elapsed = time.time() - start
        output = result.stdout.strip()

        if result.returncode != 0:
            return {"ok": False, "error": f"exit code {result.returncode}", "time": elapsed}
        if not output:
            return {"ok": False, "error": "empty response", "time": elapsed}
        if "error" in output.lower() and "success" not in output.lower():
            return {"ok": False, "error": output[:100], "time": elapsed}

        return {"ok": True, "url": output[:150], "time": elapsed}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout (15s)", "time": 15}
    except Exception as e:
        return {"ok": False, "error": str(e), "time": 0}


def main():
    print("Testing file sharing services...\n")

    # Create test file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("test file for service check")
        test_file = f.name

    results = []
    for service in SERVICES:
        print(f"Testing {service['name']}...", end=" ", flush=True)
        result = test_service(service, test_file)
        result["name"] = service["name"]
        result["expiry"] = service["expiry"]
        result["limit"] = service["limit"]
        results.append(result)

        if result["ok"]:
            print(f"OK ({result['time']:.1f}s)")
        else:
            print(f"FAILED: {result['error']}")

    # Clean up
    Path(test_file).unlink(missing_ok=True)

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)

    available = [r for r in results if r["ok"]]
    if available:
        print("\nAvailable services:")
        for r in available:
            print(f"  + {r['name']:15} {r['expiry']:12} {r['limit']:8} ({r['time']:.1f}s)")

    unavailable = [r for r in results if not r["ok"]]
    if unavailable:
        print("\nUnavailable services:")
        for r in unavailable:
            print(f"  - {r['name']:15} {r['error']}")

    if not available:
        print("\nNo services available from current network!")
        return 1

    print(f"\nRecommended: {available[0]['name']}")
    return 0


if __name__ == "__main__":
    exit(main())
