#!/usr/bin/env python3
"""Download public UsedSurf product-gallery images for local shot labeling."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import subprocess
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "shot-labels"
BASE = "https://usedsurf.com/used-surfboards/"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: set[str] = set()
        self.images: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "a" and values.get("href"):
            self.links.add(values["href"])
        if tag.lower() in {"img", "source"}:
            for key in ("src", "data-src", "data-image", "srcset", "data-srcset"):
                if values.get(key):
                    self.images.extend(part.strip().split(" ", 1)[0] for part in values[key].split(","))


def fetch(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(3):
        request = Request(url, headers={"User-Agent": "UsedSurf-local-shot-labeler/1.0"})
        try:
            with urlopen(request, timeout=30, context=ssl.create_default_context()) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            try:
                # UsedSurf is behind Cloudflare and may reject Python's TLS
                # fingerprint. macOS ships curl, which receives the public
                # page normally. No cookies or credentials are required.
                result = subprocess.run(["curl", "-fsSL", "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131 Safari/537.36", url], check=True, capture_output=True)
                return result.stdout
            except Exception as curl_error:
                last_error = curl_error
                if attempt < 2:
                    time.sleep(1.0 + attempt)
    raise RuntimeError(f"download failed: {url}: {last_error}")


def parse(raw: bytes) -> LinkParser:
    parser = LinkParser()
    parser.feed(raw.decode("utf-8", errors="ignore"))
    return parser


def product_links(raw: bytes, page_url: str) -> list[str]:
    parser = parse(raw)
    found = []
    for href in parser.links:
        absolute = urljoin(page_url, href).split("#", 1)[0].rstrip("/")
        parsed = urlparse(absolute)
        if parsed.netloc.endswith("usedsurf.com") and re.search(r"used-surfboard[-/]", parsed.path, re.I):
            found.append(absolute)
    return sorted(set(found))


def gallery_links(raw: bytes, page_url: str) -> list[str]:
    parser = parse(raw)
    found = []
    for value in parser.images:
        absolute = urljoin(page_url, value).split("?", 1)[0]
        parsed = urlparse(absolute)
        if not (parsed.netloc.endswith("bigcommerce.com") or parsed.netloc.endswith("usedsurf.com")):
            continue
        if "/products/" not in parsed.path:
            continue
        if not re.search(r"\.(?:jpe?g|png|webp)$", parsed.path, re.I):
            continue
        if any(token in parsed.path.lower() for token in ("logo", "icon", "sprite", "favicon")):
            continue
        # Replace the storefront's responsive-size placeholder with a stable
        # large image size before downloading.
        absolute = re.sub(r"/images/stencil/[^/]+/products/", "/images/stencil/1280x1280/products/", absolute)
        found.append(absolute)
    return list(dict.fromkeys(found))


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")[:100] or "board"


def ordered_label(position: int) -> str | None:
    if 1 <= position <= 4:
        return "full_board"
    if position == 5:
        return "side_profile"
    if position == 6:
        return "fin_detail"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--max-boards", type=int, default=100)
    parser.add_argument("--max-photos-per-board", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.35)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"version": 1, "cases": []}
    existing = {case["source_url"]: case for case in manifest["cases"]}
    errors = list(manifest.get("errors", []))
    # The store's gallery contract is stable: four board views, then rail,
    # then an optional fin/detail image. Treat this as weak/order-derived
    # supervision, never as a human correction.
    for case in existing.values():
        position_match = re.search(r"/(\d{2})-[^/]+$", case.get("local_path", ""))
        position = int(position_match.group(1)) if position_match else int(case.get("gallery_position") or 0)
        case["gallery_position"] = position
        if case.get("label_source") != "human" and case.get("status") != "labeled":
            case["label"] = ordered_label(position)
            case["label_source"] = "gallery_order"
            case["status"] = "auto_labeled" if case["label"] else "unlabeled"
    links: list[str] = []
    for page in range(1, args.pages + 1):
        url = BASE if page == 1 else f"{BASE}?page={page}"
        print(f"catalog {page}/{args.pages}: {url}")
        links.extend(product_links(fetch(url), url))
        time.sleep(args.delay)
    for product_url in list(dict.fromkeys(links))[: args.max_boards]:
        print(f"product: {product_url}")
        try:
            html = fetch(product_url)
        except Exception as exc:
            print(f"  skipped product page: {exc}")
            errors.append({"product_url": product_url, "error": str(exc)})
            continue
        slug = safe(urlparse(product_url).path.rstrip("/").split("/")[-1])
        product_id = re.search(rb'name=["\']product_id["\']\s+value=["\'](\d+)', html, re.I)
        raw_gallery = gallery_links(html, product_url)
        if product_id:
            product_token = f"/products/{product_id.group(1).decode()}/"
            raw_gallery = [url for url in raw_gallery if product_token in url]
        for position, image_url in enumerate(raw_gallery[: args.max_photos_per_board], start=1):
            if image_url in existing:
                continue
            try:
                content = fetch(image_url)
            except Exception:
                # Some BigCommerce image sizes are missing even though the
                # original product asset exists. Retry that one image at the
                # original-size endpoint before marking it failed.
                fallback_url = re.sub(r"/images/stencil/[^/]+/products/", "/images/stencil/original/products/", image_url)
                try:
                    content = fetch(fallback_url)
                    image_url = fallback_url
                except Exception as exc:
                    print(f"  skipped image: {image_url} ({exc})")
                    errors.append({"product_url": product_url, "source_url": image_url, "error": str(exc)})
                    continue
            extension = ".jpg" if re.search(r"\.jpe?g$", image_url, re.I) else ".png"
            digest = hashlib.sha256(content).hexdigest()[:12]
            relative = Path("images") / slug / f"{position:02d}-{digest}{extension}"
            target = OUT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            label = ordered_label(position)
            existing[image_url] = {"id": f"{slug}-{position:02d}-{digest}", "board_key": slug, "gallery_position": position, "source_url": image_url, "product_url": product_url, "local_path": str(Path("data") / "shot-labels" / relative), "label": label, "label_source": "gallery_order" if label else None, "status": "auto_labeled" if label else "unlabeled"}
            time.sleep(args.delay)
        time.sleep(args.delay)
    manifest["cases"] = list(existing.values())
    manifest["errors"] = errors[-200:]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {len(manifest['cases'])} images in {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
