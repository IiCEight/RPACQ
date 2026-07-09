from loguru import logger
import sys


def setUpLogger(level=str):
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> "
            "{name: <8} "
            "{function: <8} "
            "{line: <3} "
            "<level>{message}</level>"
        ),
    )
