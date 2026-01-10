"""
Query command for searching chat messages.

Usage:
    /query_chat content="keyword" [user=123456] [after=2025-01-01] [before=2025-01-10] [limit=20]
    /query_chat regex="pattern.*" [user=123456] [after=7d] [limit=20]

In group chat: Only queries current group's messages, available to all users.
In private chat: Only available to superusers, must specify group=<group_id>.
"""

import re
import time as time_module
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

from nonebot import on_command, logger
from nonebot.adapters.onebot.v11 import (
    GroupMessageEvent,
    PrivateMessageEvent,
)
from nonebot.params import CommandArg
from nonebot.adapters.onebot.v11 import Message
from nonebot_plugin_uninfo import QryItrface, SceneType

from .models import ChatMessage
from .config import SUPERUSERS
from .cache import (
    CacheEntry,
    CacheResult,
    HotColdCache,
    QueryFilter as CacheQueryFilter,
    get_query_cache,
    MAX_MESSAGES_PER_ENTRY,
)


# UTC+8 timezone
TZ_UTC8 = timezone(timedelta(hours=8))

# Default and max limits
DEFAULT_LIMIT = 20
MAX_LIMIT = 200
DEFAULT_AFTER_DAYS = 30  # Default to 30 days ago


def parse_query_args(arg_str: str) -> dict[str, str]:
    """
    Parse query arguments in key=value or key="value" format.

    Examples:
        content="hello world" user=123 after=7d
        regex="test.*" limit=100
    """
    result = {}
    # Match: key=value or key="value" or key='value'
    pattern = r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))'

    for match in re.finditer(pattern, arg_str):
        key = match.group(1)
        # Value is in group 2 (double quoted), 3 (single quoted), or 4 (unquoted)
        value = match.group(2) or match.group(3) or match.group(4)
        result[key] = value

    return result


def parse_time(time_str: str) -> Optional[int]:
    """
    Parse time string to unix timestamp.

    Supports:
        - Relative: 7d, 24h, 30m (days, hours, minutes ago)
        - Absolute: 2025-01-01, 2025-01-01T12:00, 2025-01-01 12:00
    """
    if not time_str:
        return None

    # Relative time patterns
    relative_pattern = r"^(\d+)([dhm])$"
    if match := re.match(relative_pattern, time_str.lower()):
        amount = int(match.group(1))
        unit = match.group(2)
        now = datetime.now(TZ_UTC8)

        if unit == "d":
            delta = timedelta(days=amount)
        elif unit == "h":
            delta = timedelta(hours=amount)
        elif unit == "m":
            delta = timedelta(minutes=amount)
        else:
            return None

        return int((now - delta).timestamp())

    # Absolute time patterns
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            # Assume input is in UTC+8
            dt = dt.replace(tzinfo=TZ_UTC8)
            return int(dt.timestamp())
        except ValueError:
            continue

    return None


def format_timestamp(ts: int) -> str:
    """Format unix timestamp to readable string in UTC+8."""
    dt = datetime.fromtimestamp(ts, tz=TZ_UTC8)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate_message(msg: str, max_len: int = 100) -> str:
    """Truncate message if too long."""
    # Remove newlines for display
    msg = msg.replace("\n", " ").replace("\r", "")
    if len(msg) > max_len:
        return msg[: max_len - 3] + "..."
    return msg


class QueryFilter:
    """Represents parsed query filter conditions."""

    def __init__(self):
        self.content: Optional[str] = None
        self.regex: Optional[str] = None
        self.user_id: Optional[int] = None
        self.group_id: Optional[int] = None
        self.time_after: Optional[int] = None
        self.time_before: Optional[int] = None
        self.limit: int = DEFAULT_LIMIT

    @classmethod
    def from_args(cls, args: dict[str, str]) -> "QueryFilter":
        """Create QueryFilter from parsed arguments."""
        f = cls()

        if "content" in args:
            f.content = args["content"]

        if "regex" in args:
            f.regex = args["regex"]
            # Validate regex
            try:
                re.compile(f.regex)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")

        if "user" in args:
            try:
                f.user_id = int(args["user"])
            except ValueError:
                raise ValueError(f"Invalid user ID: {args['user']}")

        if "group" in args:
            try:
                f.group_id = int(args["group"])
            except ValueError:
                raise ValueError(f"Invalid group ID: {args['group']}")

        if "after" in args:
            f.time_after = parse_time(args["after"])
            if f.time_after is None:
                raise ValueError(f"Invalid time format for 'after': {args['after']}")
        else:
            # Default to 30 days ago
            f.time_after = int(time_module.time()) - DEFAULT_AFTER_DAYS * 24 * 3600

        if "before" in args:
            f.time_before = parse_time(args["before"])
            if f.time_before is None:
                raise ValueError(f"Invalid time format for 'before': {args['before']}")

        if "limit" in args:
            try:
                f.limit = min(int(args["limit"]), MAX_LIMIT)
                if f.limit <= 0:
                    f.limit = DEFAULT_LIMIT
            except ValueError:
                f.limit = DEFAULT_LIMIT

        return f

    def has_message_filter(self) -> bool:
        """Check if at least one message filter (content or regex) is set."""
        return self.content is not None or self.regex is not None

    def format_conditions(self) -> str:
        """Format filter conditions for display."""
        parts = []

        if self.content:
            parts.append(f'content="{self.content}"')
        if self.regex:
            parts.append(f'regex="{self.regex}"')
        if self.user_id:
            parts.append(f"user={self.user_id}")
        if self.group_id:
            parts.append(f"group={self.group_id}")
        if self.time_after:
            parts.append(f"after={format_timestamp(self.time_after)}")
        if self.time_before:
            parts.append(f"before={format_timestamp(self.time_before)}")
        parts.append(f"limit={self.limit}")

        return " | ".join(parts)

    def to_cache_filter(self) -> CacheQueryFilter:
        """Convert to cache module's QueryFilter."""
        return CacheQueryFilter(
            group_id=self.group_id,
            user_id=self.user_id,
            content=self.content,
            regex=self.regex,
            time_after=self.time_after,
            time_before=self.time_before,
            limit=self.limit,
        )


async def execute_query_raw(
    qf: QueryFilter,
    time_after_override: Optional[int] = None,
    time_before_override: Optional[int] = None,
) -> list[ChatMessage]:
    """
    Execute query against database without caching.
    Returns all matching messages (up to MAX_MESSAGES_PER_ENTRY).

    Args:
        qf: Query filter
        time_after_override: Override time_after for incremental queries
        time_before_override: Override time_before for incremental queries
    """
    # Build base query
    base_query = ChatMessage.filter(group_id=qf.group_id)

    # Apply time filters (with optional overrides)
    time_after = (
        time_after_override if time_after_override is not None else qf.time_after
    )
    time_before = (
        time_before_override if time_before_override is not None else qf.time_before
    )

    if time_after:
        base_query = base_query.filter(time__gte=time_after)
    if time_before:
        base_query = base_query.filter(time__lte=time_before)

    # Apply user filter
    if qf.user_id:
        base_query = base_query.filter(user_id=qf.user_id)

    # Order by time descending
    base_query = base_query.order_by("-time")

    # If no content/regex filter, simple query
    if not qf.content and not qf.regex:
        query = base_query.limit(MAX_MESSAGES_PER_ENTRY)
        logger.info(f"Query SQL: {query.sql(params_inline=True)}")
        return await query

    # With content/regex filter, use batch iteration
    BATCH_SIZE = 5000
    MAX_BATCHES = 20  # Safety limit: max 100k records scanned

    all_matches: list[ChatMessage] = []
    regex_pattern = re.compile(qf.regex) if qf.regex else None
    offset = 0

    for _ in range(MAX_BATCHES):
        query = base_query.offset(offset).limit(BATCH_SIZE)
        logger.info(f"Query SQL: {query.sql(params_inline=True)}")

        batch = await query
        if not batch:
            break  # No more data

        for msg in batch:
            # Use plain_text only for matching
            text = msg.plain_text

            if qf.content and qf.content not in text:
                continue

            if regex_pattern and not regex_pattern.search(text):
                continue

            all_matches.append(msg)

            # Stop if we have enough
            if len(all_matches) >= MAX_MESSAGES_PER_ENTRY:
                return all_matches

        offset += BATCH_SIZE

    return all_matches


async def execute_query_with_cache(
    qf: QueryFilter,
) -> tuple[list[ChatMessage], int, bool, Optional[int]]:
    """
    Execute query with cache support.

    Returns: (messages, total_count, is_fuzzy, cache_percent)
        - messages: Limited by qf.limit
        - total_count: Total matching messages
        - is_fuzzy: True if result is from fuzzy time match (display ~)
        - cache_percent: None if no cache, 100 if full cache hit, 0-99 if partial
    """
    cache = get_query_cache()
    cache_filter = qf.to_cache_filter()

    # Try cache lookup
    cache_result = cache.get(cache_filter)

    if cache_result is not None:
        if not cache_result.needs_incremental:
            # Cache hit (exact or fuzzy) - 100% cached
            logger.info(f"Cache hit (fuzzy={cache_result.is_fuzzy})")
            return (
                cache_result.messages[: qf.limit],
                cache_result.total_count,
                cache_result.is_fuzzy,
                100,  # Full cache hit
            )

        # Incremental query needed
        logger.info(
            f"Cache partial hit, incremental query for: {cache_result.missing_ranges}"
        )
        return await _execute_incremental_query(qf, cache_result, cache)

    # Cache miss, execute full query
    logger.info("Cache miss, executing full query")
    all_messages = await execute_query_raw(qf)

    # Store in cache
    entry = CacheEntry(
        group_id=qf.group_id,  # type: ignore
        user_id=qf.user_id,
        content=qf.content,
        regex=qf.regex,
        time_after=qf.time_after,
        time_before=qf.time_before,
        messages=all_messages,
        total_count=len(all_messages),
        cached_at=time_module.time(),
    )
    cache.put(entry)

    return all_messages[: qf.limit], len(all_messages), False, None  # No cache


async def _execute_incremental_query(
    qf: QueryFilter,
    cache_result: CacheResult,
    cache: HotColdCache,
) -> tuple[list[ChatMessage], int, bool, int]:
    """
    Execute incremental query and merge with cached results.
    Returns cache_percent based on how much was from cache.
    """
    # Get cached messages
    cached_messages = list(cache_result.messages)
    cached_count = len(cached_messages)

    # Query missing ranges
    incremental_messages: list[ChatMessage] = []
    for start, end in cache_result.missing_ranges:
        batch = await execute_query_raw(
            qf, time_after_override=start, time_before_override=end
        )
        incremental_messages.extend(batch)

    # Merge and deduplicate by message_id
    all_messages = cached_messages + incremental_messages
    seen_ids: set[int] = set()
    unique_messages: list[ChatMessage] = []

    for msg in all_messages:
        if msg.message_id not in seen_ids:
            seen_ids.add(msg.message_id)
            unique_messages.append(msg)

    # Sort by time descending
    unique_messages.sort(key=lambda m: m.time, reverse=True)

    # Limit to max cache size
    if len(unique_messages) > MAX_MESSAGES_PER_ENTRY:
        unique_messages = unique_messages[:MAX_MESSAGES_PER_ENTRY]

    # Calculate cache percentage
    total_count = len(unique_messages)
    cache_percent = int(cached_count * 100 / total_count) if total_count > 0 else 0

    # Calculate merged time range
    old_entry = cache_result.cache_entry
    if old_entry:
        merged_time_after = min(
            old_entry.time_after or float("inf"),
            qf.time_after or float("inf"),
        )
        merged_time_before = max(
            old_entry.time_before or 0,
            qf.time_before or 0,
        )

        # Convert to int (handle float('inf'))
        merged_time_after = (
            int(merged_time_after) if merged_time_after != float("inf") else None
        )
        merged_time_before = (
            int(merged_time_before) if merged_time_before != 0 else None
        )

        # Update cache with merged entry
        old_key = old_entry.make_key()
        new_entry = CacheEntry(
            group_id=qf.group_id,  # type: ignore
            user_id=qf.user_id,
            content=qf.content,
            regex=qf.regex,
            time_after=merged_time_after,
            time_before=merged_time_before,
            messages=unique_messages,
            total_count=len(unique_messages),
            cached_at=time_module.time(),
        )
        cache.update_entry(old_key, new_entry)

    return unique_messages[: qf.limit], len(unique_messages), False, cache_percent


async def format_results(
    qf: QueryFilter,
    messages: list[ChatMessage],
    total_count: int,
    show_user: bool = True,
    user_names: dict[int, str] | None = None,
    is_fuzzy: bool = False,
    cache_percent: Optional[int] = None,
) -> str:
    """Format query results for display."""
    lines = [f"Query: {qf.format_conditions()}"]

    # Format count line based on cache status
    count_prefix = "~" if is_fuzzy else ""
    if cache_percent is not None:
        if cache_percent == 100:
            # Full cache hit
            lines.append(
                f"Showing {len(messages)}/{count_prefix}{total_count} (cached)"
            )
        else:
            # Partial cache hit
            lines.append(
                f"Showing {len(messages)}/{count_prefix}{total_count} ({cache_percent}% cached)"
            )
    else:
        # No cache
        lines.append(f"Showing {len(messages)}/{total_count}")

    lines.append("-" * 40)

    for msg in messages:
        time_str = format_timestamp(msg.time)
        text = truncate_message(msg.plain_text)

        if show_user:
            # Use nickname if available, otherwise fallback to user_id
            user_display = (
                user_names.get(msg.user_id, str(msg.user_id))
                if user_names
                else str(msg.user_id)
            )
            lines.append(f"[{time_str}] {user_display}: {text}")
        else:
            lines.append(f"[{time_str}] {text}")

    return "\n".join(lines)


async def get_group_member_names(
    interface: QryItrface, group_id: int, user_ids: set[int]
) -> dict[int, str]:
    """Fetch member nicknames for given user IDs in a group."""
    user_names: dict[int, str] = {}

    try:
        members = await interface.get_members(SceneType.GROUP, str(group_id))
        for member in members:
            uid = int(member.user.id)
            if uid in user_ids:
                # Prefer member nick, then user name, then user id
                name = member.nick or member.user.name or str(uid)
                user_names[uid] = name
    except Exception as e:
        logger.warning(f"Failed to fetch member names: {e}")

    return user_names


# Create command handler
query_chat = on_command("query_chat", priority=10, block=True)


@query_chat.handle()
async def handle_query(
    event: Union[GroupMessageEvent, PrivateMessageEvent],
    args: Message = CommandArg(),
    interface: QryItrface = None,
):
    """Handle query_chat command."""
    arg_str = args.extract_plain_text().strip()

    # Parse arguments
    try:
        parsed_args = parse_query_args(arg_str)
        qf = QueryFilter.from_args(parsed_args)
    except ValueError as e:
        await query_chat.finish(f"Parameter error: {e}")
        return

    # Check if at least content or regex is provided
    if not qf.has_message_filter():
        await query_chat.finish(
            "Error: Must specify at least 'content' or 'regex' parameter.\n"
            'Usage: /query_chat content="keyword" [user=123] [after=7d] [limit=20]\n'
            '       /query_chat regex="pattern.*" [user=123]'
        )
        return

    # Permission and group_id handling
    is_group = isinstance(event, GroupMessageEvent)
    if is_group:
        # In group chat: use current group, available to all
        qf.group_id = event.group_id
        show_user = qf.user_id is None  # Show user if not filtering by user
    else:
        # In private chat: only superusers, must specify group
        if event.user_id not in SUPERUSERS:
            await query_chat.finish(
                "This command is only available to superusers in private chat."
            )
            return

        if qf.group_id is None:
            await query_chat.finish(
                "Error: Must specify 'group' parameter in private chat.\n"
                'Usage: /query_chat content="keyword" group=123456789'
            )
            return

        show_user = qf.user_id is None

    # Execute query with cache
    try:
        messages, total_count, is_fuzzy, cache_percent = await execute_query_with_cache(
            qf
        )
    except Exception as e:
        logger.exception("Query execution failed")
        await query_chat.finish(f"Query failed: {e}")
        return

    if not messages:
        await query_chat.finish(
            f"Query: {qf.format_conditions()}\n\nNo messages found."
        )
        return

    # Fetch user nicknames for group chat
    user_names: dict[int, str] | None = None
    if is_group and show_user and interface and qf.group_id:
        user_ids = {msg.user_id for msg in messages}
        user_names = await get_group_member_names(interface, qf.group_id, user_ids)

    # Format and send results
    result_text = await format_results(
        qf,
        messages,
        total_count,
        show_user=show_user,
        user_names=user_names,
        is_fuzzy=is_fuzzy,
        cache_percent=cache_percent,
    )

    # Truncate if too long (QQ message limit)
    if len(result_text) > 4000:
        result_text = result_text[:3997] + "..."

    await query_chat.finish(result_text)
