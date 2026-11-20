"""Semantic cache for an FAQ bot.

Answer repeat questions from a cache when the *meaning* matches, call the LLM only on a miss,
and prove it with a hit rate, a false-hit count and a latency number.
"""

__version__ = "1.0.0"
