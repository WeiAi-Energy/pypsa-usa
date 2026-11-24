"""
Download one workflow input from the URL in ``snakemake.params.url``, resuming broken transfers.

Large files (the ~3.6 GB line ratings on Zenodo) are prone to connections closed
mid-transfer. The data goes to ``<output>.part``; after a broken connection, a
network error or a timeout the download continues from the bytes already there with
an HTTP ``Range`` request, and the file is renamed to ``<output>`` only when its size
matches the size the server announced. A server that ignores ``Range`` (status 200)
makes the download start over.
"""

import logging
import time
import urllib.error
import urllib.request
from http.client import HTTPException
from pathlib import Path

from _helpers import configure_logging

logger = logging.getLogger(__name__)

CHUNK = 1 << 20  # bytes read and written at a time
ATTEMPTS = 20  # connections tried before giving up
TIMEOUT = 60  # seconds without data before a connection counts as broken
RETRY_ERRORS = (urllib.error.URLError, HTTPException, ConnectionError, TimeoutError, OSError)


def _total_size(response, offset: int) -> int | None:
    """Full size of the remote file from Content-Range (``bytes a-b/total``) or Content-Length."""
    content_range = response.headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1].strip()
        if total.isdigit():
            return int(total)
    length = response.headers.get("Content-Length")
    if length is not None and length.isdigit():
        return int(length) + offset
    return None


def download(
    url: str,
    output: str | Path,
    attempts: int = ATTEMPTS,
    timeout: float = TIMEOUT,
    chunk: int = CHUNK,
    backoff: float = 5.0,
) -> Path:
    """Download ``url`` to ``output``, resuming from ``<output>.part`` after broken connections."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_name(output.name + ".part")
    total = None
    last_logged = -1

    for attempt in range(1, attempts + 1):
        offset = part.stat().st_size if part.exists() else 0
        if total is not None and offset == total:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
                if offset and response.status != 206:
                    logger.info("Server ignored the range request (status %d); starting over.", response.status)
                    offset = 0
                total = _total_size(response, offset)
                if offset:
                    logger.info("Resuming at %.0f MB of %s.", offset / 1e6, f"{total / 1e6:.0f} MB" if total else "?")
                with open(part, "ab" if offset else "wb") as fh:
                    while block := response.read(chunk):
                        fh.write(block)
                        offset += len(block)
                        if total:
                            percent = int(100 * offset / total)
                            if percent // 5 > last_logged:
                                last_logged = percent // 5
                                logger.info("  %d%% (%.0f of %.0f MB)", percent, offset / 1e6, total / 1e6)
            if total is None or offset == total:
                break
            logger.warning("Connection closed at %.0f of %.0f MB (attempt %d).", offset / 1e6, total / 1e6, attempt)
        except urllib.error.HTTPError as err:
            if err.code == 416:  # range starts at or past the end; the server names the size (bytes */total)
                announced = err.headers.get("Content-Range", "").rsplit("/", 1)[-1].strip()
                if announced.isdigit() and offset == int(announced):
                    total = offset
                    break  # the part file is already complete
                logger.info("Part file of %d bytes does not match the remote file; starting over.", offset)
                part.unlink()
                continue
            if err.code < 500 and err.code not in (408, 429):
                raise
            logger.warning("HTTP %d at %.0f MB (attempt %d): %s", err.code, offset / 1e6, attempt, err)
        except RETRY_ERRORS as err:
            logger.warning("Download broken at %.0f MB (attempt %d): %r", offset / 1e6, attempt, err)
        if attempt < attempts:
            time.sleep(min(backoff * attempt, 60))
    else:
        raise RuntimeError(f"Could not download {url} in {attempts} attempts ({offset} bytes in {part}).")

    size = part.stat().st_size
    if total is not None and size != total:
        raise RuntimeError(f"{part} has {size} bytes, the server announced {total}.")
    part.replace(output)
    logger.info("Downloaded %s (%.0f MB).", output, size / 1e6)
    return output


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "retrieve_reeds_vre_file",
            reeds_vre_tech="solar",
            reeds_vre_file="sc_upv_reference.csv",
        )
    configure_logging(snakemake)
    logger.info("Downloading %s to %s.", snakemake.params.url, snakemake.output[0])
    download(snakemake.params.url, snakemake.output[0])
