import logging
from typing import List, Dict

logger = logging.getLogger("graph_writer")


def write(user_id: str, beliefs: List[Dict]):
    logger.info("Persisting %d beliefs for user %s", len(beliefs), user_id)
