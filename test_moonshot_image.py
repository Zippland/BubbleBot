#!/usr/bin/env python3
"""Test Moonshot API image support directly (bypassing LiteLLM)."""

import asyncio
import base64
import os
import sys
from pathlib import Path

import httpx


async def test_moonshot_image(image_path: str):
    """Test sending an image to Moonshot API directly."""
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        print("ERROR: MOONSHOT_API_KEY not set")
        return

    # Read and encode image
    p = Path(image_path)
    if not p.is_file():
        print(f"ERROR: File not found: {image_path}")
        return

    file_bytes = p.read_bytes()
    b64 = base64.b64encode(file_bytes).decode()

    # Determine mime type
    ext = p.suffix.lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext.lstrip("."), "image/jpeg")

    print(f"Image: {image_path}")
    print(f"Size: {len(file_bytes)} bytes")
    print(f"Base64 length: {len(b64)}")
    print(f"MIME: {mime}")
    print()

    # Build request
    url = "https://api.moonshot.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "kimi-k2.5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}
                    },
                    {
                        "type": "text",
                        "text": "What do you see in this image? Describe it briefly."
                    }
                ]
            }
        ],
        "max_tokens": 500,
        "temperature": 1.0,
    }

    print("Sending request to Moonshot API...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        print(f"\nResponse:\n{content}")
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_moonshot_image.py <image_path>")
        sys.exit(1)

    asyncio.run(test_moonshot_image(sys.argv[1]))
