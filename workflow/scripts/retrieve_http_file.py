"""Download a workflow input from a configured HTTP URL."""

import logging
from pathlib import Path

from _helpers import configure_logging, progress_retrieve

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    configure_logging(snakemake)
    output = Path(snakemake.output[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading workflow input from %s.", snakemake.params.url)
    progress_retrieve(snakemake.params.url, output)
