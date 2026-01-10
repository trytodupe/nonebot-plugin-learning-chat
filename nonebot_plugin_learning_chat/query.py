"""
Query command for searching chat messages.

Usage:
    /query_chat content="keyword" [user=123456] [after=2025-01-01] [before=2025-01-10] [limit=50]
    /query_chat regex="pattern.*" [user=123456] [after=7d] [limit=50]

In group chat: Only queries current group's messages, available to all users.
In private chat: Only available to superusers, must specify group=<group_id>.
"""

import re
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


# UTC+8 timezone
TZ_UTC8 = timezone(timedelta(hours=8))

# Default and max limits
DEFAULT_LIMIT = 50
MAX_LIMIT = 200


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


async def execute_query(qf: QueryFilter) -> tuple[list[ChatMessage], int]:
    """
    Execute query against database.
    Returns (messages limited by qf.limit, total_count of all matches).
    """
    # Build base query
    base_query = ChatMessage.filter(group_id=qf.group_id)

    # Apply time filters
    if qf.time_after:
        base_query = base_query.filter(time__gte=qf.time_after)
    if qf.time_before:
        base_query = base_query.filter(time__lte=qf.time_before)

    # Apply user filter
    if qf.user_id:
        base_query = base_query.filter(user_id=qf.user_id)

    # Order by time descending
    base_query = base_query.order_by("-time")

    # If no content/regex filter, simple query with limit
    if not qf.content and not qf.regex:
        # Get total count
        total_count = await base_query.count()
        query = base_query.limit(qf.limit)
        logger.info(f"Query SQL: {query.sql(params_inline=True)}")
        messages = await query
        return messages, total_count

    # With content/regex filter, use batch iteration to find ALL matches
    BATCH_SIZE = 5000
    MAX_BATCHES = 20  # Safety limit: max 100k records scanned

    all_matches = []
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

        offset += BATCH_SIZE

    total_count = len(all_matches)
    return all_matches[: qf.limit], total_count


async def format_results(
    qf: QueryFilter,
    messages: list[ChatMessage],
    total_count: int,
    show_user: bool = True,
    user_names: dict[int, str] | None = None,
) -> str:
    """Format query results for display."""
    lines = [f"Query: {qf.format_conditions()}"]
    lines.append(f"Found: {total_count} messages (showing {len(messages)})")
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
            'Usage: /query_chat content="keyword" [user=123] [after=7d] [limit=50]\n'
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

    # Execute query
    try:
        messages, total_count = await execute_query(qf)
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
        qf, messages, total_count, show_user=show_user, user_names=user_names
    )

    # Truncate if too long (QQ message limit)
    if len(result_text) > 4000:
        result_text = result_text[:3997] + "..."

    await query_chat.finish(result_text)
