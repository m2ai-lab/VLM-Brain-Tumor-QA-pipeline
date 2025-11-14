from __future__ import annotations
import argparse
from asyncio.log import logger
import json
import logging
import pathlib
import sys
from typing import Any, Dict, Optional
from PIL import Image

#!/usr/bin/env python3
"""
Main entrypoint for QA-BrainTumor-VLM-UCSF project.

This file provides a robust CLI for running a Visual-Language QA pipeline
on a single image + question.
"""


# Minimal logging setup so each module we can have specialized outputs as needed
# Call logger = logging.getLogger(__name__) in modules to get module-specific loggers
# Example usage:
#   import logger
#   logger.info("Converting DICOM files to NIfTI format...")
#   logger.debug("20 files converted...")
def setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# Build the argument parser for the CLI
# This allows us to determine what settings we will run the pipeline with taking out the hardcoding for some parts
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QA pipeline entrypoint for brain tumor VLM project")
    p.add_argument("image", type=pathlib.Path, help="Path to input image")
    p.add_argument("question", type=str, help="Natural language question about the image")
    p.add_argument("--model", type=str, default="default", help="Model name (module under models/ or 'default')")
    p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("output.json"), help="Output JSON path")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p


# Main function to run the pipeline
# This function sets up the pipeline and handles exceptions also serves as a base to where all the other modules connect to
def main(argv: Optional[list[str]] = None) -> int:
    """
    Main function. Returns exit code (0 on success).
    """
    args = build_arg_parser().parse_args(argv)
    setup_logging(args.debug)

    logging.info("Image: %s", args.image)
    logging.info("Question: %s", args.question)
    logging.info("Model: %s", args.model)

    try:
       
        return 0
    except Exception as e:
        logging.exception("Failed to run QA pipeline: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1



# Entry point for script
if __name__ == "__main__":
    raise SystemExit(main())