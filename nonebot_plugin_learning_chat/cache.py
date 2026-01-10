"""
Query cache module with hot (memory) + cold (file) two-tier LRU cache.

Features:
- Hot cache: 10 entries in memory, fastest access
- Cold cache: 50 entries persisted to file
- Weak matching: allows cache reuse with stricter query conditions
- Incremental query: reuse partial cache when time ranges overlap >40%
- Max 1000 messages per cache entry
"""

import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from nonebot import logger

if TYPE_CHECKING:
    from .models import ChatMessage


# Constants
HOT_CACHE_SIZE = 10
COLD_CACHE_SIZE = 50
MAX_MESSAGES_PER_ENTRY = 1000
TIME_FUZZY_WINDOW = 300  # 5 minutes in seconds
OVERLAP_THRESHOLD = 0.4  # 40% overlap required for incremental query


class CacheMatchType(Enum):
    """Type of cache match result."""

    NO_MATCH = 0  # Cache cannot be used
    EXACT_MATCH = 1  # Cache fully covers query
    FUZZY_TIME = 2  # Cache matches with time fuzzy window
    PARTIAL_OVERLAP = 3  # Time ranges partially overlap, needs incremental query


@dataclass
class CacheEntry:
    """Single cache entry storing query results."""

    group_id: int
    user_id: Optional[int]
    content: Optional[str]
    regex: Optional[str]
    time_after: Optional[int]
    time_before: Optional[int]

    messages: list["ChatMessage"] = field(default_factory=list)
    total_count: int = 0
    cached_at: float = field(default_factory=time.time)

    def make_key(self) -> str:
        """Generate cache key (without limit)."""
        return f"{self.group_id}:{self.user_id}:{self.content}:{self.regex}:{self.time_after}:{self.time_before}"

    def copy_filter_only(self) -> "CacheEntry":
        """Create a copy with only filter fields (no messages)."""
        return CacheEntry(
            group_id=self.group_id,
            user_id=self.user_id,
            content=self.content,
            regex=self.regex,
            time_after=self.time_after,
            time_before=self.time_before,
        )


@dataclass
class QueryFilter:
    """Query filter conditions (used for cache matching)."""

    group_id: Optional[int] = None
    user_id: Optional[int] = None
    content: Optional[str] = None
    regex: Optional[str] = None
    time_after: Optional[int] = None
    time_before: Optional[int] = None
    limit: int = 20
    # Whether absolute time was explicitly specified by user
    has_absolute_time: bool = False


@dataclass
class CacheResult:
    """Result from cache lookup."""

    messages: list["ChatMessage"]
    total_count: int
    is_fuzzy: bool  # True if matched via fuzzy time window
    needs_incremental: bool  # True if incremental query needed
    missing_ranges: list[tuple[int, int]] = field(
        default_factory=list
    )  # Time ranges to query
    cache_entry: Optional[CacheEntry] = None  # Original cache entry for merging


class HotColdCache:
    """
    Two-tier LRU cache: hot (memory) + cold (file).

    Hot cache: Fast access, limited to HOT_CACHE_SIZE entries
    Cold cache: Persisted to file, limited to COLD_CACHE_SIZE entries

    When hot cache is full, oldest entries are demoted to cold cache.
    When cold cache is full, oldest entries are evicted.
    """

    def __init__(
        self,
        hot_size: int = HOT_CACHE_SIZE,
        cold_size: int = COLD_CACHE_SIZE,
        cache_file: Optional[Path] = None,
    ):
        self.hot_size = hot_size
        self.cold_size = cold_size

        self.hot_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.cold_cache: OrderedDict[str, CacheEntry] = OrderedDict()

        self.cache_file = cache_file or Path("data/learning_chat/query_cache.pkl")
        self._load_cold_from_file()

    def _load_cold_from_file(self) -> None:
        """Load cold cache from file on startup."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    self.cold_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cold_cache)} cache entries from file")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                self.cold_cache = OrderedDict()

    def _save_to_file(self) -> None:
        """Save all cache entries (hot + cold) to file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            # Merge hot and cold for persistence
            # Cold entries first, then hot (hot will overwrite if same key)
            all_entries: OrderedDict[str, CacheEntry] = OrderedDict()
            for key, entry in self.cold_cache.items():
                all_entries[key] = entry
            for key, entry in self.hot_cache.items():
                all_entries[key] = entry
            with open(self.cache_file, "wb") as f:
                pickle.dump(all_entries, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get(self, query: QueryFilter) -> Optional[CacheResult]:
        """
        Look up cache for the given query.

        Returns CacheResult if found, None if no usable cache.
        """
        # Search hot cache first
        for key, entry in list(self.hot_cache.items()):
            result = self._try_match(entry, query)
            if result is not None:
                # Move to front (most recently used)
                self.hot_cache.move_to_end(key, last=False)
                return result

        # Search cold cache
        for key, entry in list(self.cold_cache.items()):
            result = self._try_match(entry, query)
            if result is not None:
                # Promote to hot cache
                self._promote_to_hot(key, entry)
                return result

        return None

    def put(self, entry: CacheEntry) -> None:
        """
        Add or update a cache entry.

        Enforces max messages limit and LRU eviction.
        """
        # Enforce max messages limit
        if len(entry.messages) > MAX_MESSAGES_PER_ENTRY:
            entry.messages = entry.messages[:MAX_MESSAGES_PER_ENTRY]
            entry.total_count = min(entry.total_count, MAX_MESSAGES_PER_ENTRY)

        key = entry.make_key()

        # Remove from cold if exists (will be added to hot)
        self.cold_cache.pop(key, None)

        # Add to hot cache
        self.hot_cache[key] = entry
        self.hot_cache.move_to_end(key, last=False)

        # Demote oldest from hot to cold if over limit
        while len(self.hot_cache) > self.hot_size:
            old_key, old_entry = self.hot_cache.popitem(last=True)
            self.cold_cache[old_key] = old_entry
            self.cold_cache.move_to_end(old_key, last=False)

        # Evict oldest from cold if over limit
        while len(self.cold_cache) > self.cold_size:
            self.cold_cache.popitem(last=True)

        # Persist cache
        self._save_to_file()

    def update_entry(self, old_key: str, new_entry: CacheEntry) -> None:
        """
        Update an existing cache entry with new data (for incremental merge).

        Removes old entry and adds new one.
        """
        # Remove old entry from both caches
        self.hot_cache.pop(old_key, None)
        self.cold_cache.pop(old_key, None)

        # Add new entry
        self.put(new_entry)

    def _promote_to_hot(self, key: str, entry: CacheEntry) -> None:
        """Promote a cold cache entry to hot cache."""
        # Remove from cold
        self.cold_cache.pop(key, None)

        # Add to hot
        self.hot_cache[key] = entry
        self.hot_cache.move_to_end(key, last=False)

        # Demote if over limit
        while len(self.hot_cache) > self.hot_size:
            old_key, old_entry = self.hot_cache.popitem(last=True)
            self.cold_cache[old_key] = old_entry
            self.cold_cache.move_to_end(old_key, last=False)

        # Persist
        self._save_to_file()

    def _try_match(
        self, cache: CacheEntry, query: QueryFilter
    ) -> Optional[CacheResult]:
        """
        Try to match cache entry against query.

        Returns CacheResult if usable, None otherwise.
        """
        match_type = self._can_use_cache(cache, query)

        if match_type == CacheMatchType.NO_MATCH:
            return None

        if match_type == CacheMatchType.EXACT_MATCH:
            filtered = self._filter_results(cache, query)
            return CacheResult(
                messages=filtered,
                total_count=len(filtered),  # Use filtered count, not cache total
                is_fuzzy=False,
                needs_incremental=False,
            )

        if match_type == CacheMatchType.FUZZY_TIME:
            filtered = self._filter_results(cache, query)
            return CacheResult(
                messages=filtered,
                total_count=len(filtered),  # Use filtered count, not cache total
                is_fuzzy=True,
                needs_incremental=False,
            )

        if match_type == CacheMatchType.PARTIAL_OVERLAP:
            missing = self._calculate_missing_ranges(cache, query)
            if missing:
                filtered = self._filter_results(cache, query)
                return CacheResult(
                    messages=filtered,
                    total_count=len(filtered),  # Use filtered count for partial
                    is_fuzzy=False,
                    needs_incremental=True,
                    missing_ranges=missing,
                    cache_entry=cache,
                )

        return None

    def _can_use_cache(self, cache: CacheEntry, query: QueryFilter) -> CacheMatchType:
        """
        Check if cache can be used for query.

        Strong match: group_id, regex must be equal
        Weak match: cache constraints must be <= query constraints
        """
        # Strong match: group_id
        if cache.group_id != query.group_id:
            return CacheMatchType.NO_MATCH

        # Strong match: regex
        if cache.regex != query.regex:
            return CacheMatchType.NO_MATCH

        # Weak match: user_id
        # Cache with user_id can only be used if query has same user_id
        # Cache without user_id can be used for any query (will filter)
        if cache.user_id is not None and cache.user_id != query.user_id:
            return CacheMatchType.NO_MATCH

        # Weak match: content
        # Cache content must be substring of query content (or cache has no content)
        if cache.content is not None:
            if query.content is None:
                return CacheMatchType.NO_MATCH
            if cache.content not in query.content:
                return CacheMatchType.NO_MATCH

        # Time matching
        return self._check_time_match(cache, query)

    def _check_time_match(
        self, cache: CacheEntry, query: QueryFilter
    ) -> CacheMatchType:
        """
        Check time range matching between cache and query.

        Returns:
        - EXACT_MATCH: cache fully covers query time range
        - FUZZY_TIME: no absolute time specified, overlap >90%, uncovered <10min
        - PARTIAL_OVERLAP: ranges overlap >40%, can use incremental query
        - NO_MATCH: no usable overlap
        """
        # Check if cache fully covers query time range
        time_covered = True

        # time_after: cache.time_after must be <= query.time_after
        if cache.time_after is not None:
            if query.time_after is None or query.time_after < cache.time_after:
                time_covered = False

        # time_before: cache.time_before must be >= query.time_before
        if cache.time_before is not None:
            if query.time_before is None or query.time_before > cache.time_before:
                time_covered = False

        if time_covered:
            return CacheMatchType.EXACT_MATCH

        # Calculate overlap metrics for fuzzy/partial matching
        cache_start = cache.time_after or 0
        cache_end = cache.time_before or int(time.time())
        query_start = query.time_after or 0
        query_end = query.time_before or int(time.time())

        # Calculate overlap
        overlap_start = max(cache_start, query_start)
        overlap_end = min(cache_end, query_end)

        if overlap_start >= overlap_end:
            return CacheMatchType.NO_MATCH  # No overlap

        overlap_duration = overlap_end - overlap_start
        query_duration = query_end - query_start

        if query_duration <= 0:
            return CacheMatchType.NO_MATCH

        overlap_ratio = overlap_duration / query_duration
        uncovered_duration = query_duration - overlap_duration

        # FUZZY_TIME: only when no absolute time specified by user
        # Conditions: overlap >90% AND uncovered <10 minutes
        if not query.has_absolute_time:
            if overlap_ratio > 0.9 and uncovered_duration < 600:  # 10 minutes
                return CacheMatchType.FUZZY_TIME

        # PARTIAL_OVERLAP: overlap >40%, use incremental query
        if overlap_ratio >= OVERLAP_THRESHOLD:
            return CacheMatchType.PARTIAL_OVERLAP

        return CacheMatchType.NO_MATCH

    def _calculate_overlap_ratio(self, cache: CacheEntry, query: QueryFilter) -> float:
        """
        Calculate the overlap ratio between cache and query time ranges.

        Returns value between 0.0 and 1.0.
        """
        # Get effective time ranges
        cache_start = cache.time_after or 0
        cache_end = cache.time_before or int(time.time())
        query_start = query.time_after or 0
        query_end = query.time_before or int(time.time())

        # Calculate overlap
        overlap_start = max(cache_start, query_start)
        overlap_end = min(cache_end, query_end)

        if overlap_start >= overlap_end:
            return 0.0  # No overlap

        overlap_duration = overlap_end - overlap_start
        query_duration = query_end - query_start

        if query_duration <= 0:
            return 0.0

        return overlap_duration / query_duration

    def _calculate_missing_ranges(
        self, cache: CacheEntry, query: QueryFilter
    ) -> list[tuple[int, int]]:
        """
        Calculate time ranges that need to be queried incrementally.

        Returns list of (start, end) tuples.
        """
        cache_start = cache.time_after or 0
        cache_end = cache.time_before or int(time.time())
        query_start = query.time_after or 0
        query_end = query.time_before or int(time.time())

        missing = []

        # Query starts before cache
        if query_start < cache_start:
            missing.append((query_start, cache_start - 1))

        # Query ends after cache
        if query_end > cache_end:
            missing.append((cache_end + 1, query_end))

        return missing

    def _filter_results(
        self, cache: CacheEntry, query: QueryFilter
    ) -> list["ChatMessage"]:
        """
        Filter cached messages to match query conditions.

        Used when cache is less restrictive than query.
        Returns ALL matching messages (no limit applied).
        """
        results = cache.messages

        # Filter by user_id if query is more restrictive
        if query.user_id is not None and cache.user_id is None:
            results = [m for m in results if m.user_id == query.user_id]

        # Filter by content if query is more restrictive
        if query.content is not None:
            if cache.content is None or query.content != cache.content:
                results = [m for m in results if query.content in (m.plain_text or "")]

        # Filter by time range
        if query.time_after is not None:
            results = [m for m in results if m.time >= query.time_after]

        if query.time_before is not None:
            results = [m for m in results if m.time <= query.time_before]

        # Do NOT apply limit here - let caller handle it
        return results

    def clear(self) -> None:
        """Clear all cache entries."""
        self.hot_cache.clear()
        self.cold_cache.clear()
        self._save_to_file()

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "hot_count": len(self.hot_cache),
            "cold_count": len(self.cold_cache),
            "hot_keys": list(self.hot_cache.keys()),
            "cold_keys": list(self.cold_cache.keys()),
        }


# Global cache instance
_query_cache: Optional[HotColdCache] = None


def get_query_cache() -> HotColdCache:
    """Get or create the global query cache instance."""
    global _query_cache
    if _query_cache is None:
        _query_cache = HotColdCache()
    return _query_cache
