"""Remote saved groups (growthbook/growthbook#7277).

A remote saved group keeps its members outside the payload, which only says
which attribute the group is about: ``{"type": "remote", "attributeKey": ...}``.
The app resolves which of them a user is in, and evaluation reads the ids from
the reserved ``__gb_remoteGroupIds`` attribute.
"""

import math
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, TypedDict

# The attribute evaluation reads a user's remote saved groups from. Internal
# attributes start with `__gb_`, matching GrowthBook's back end.
REMOTE_GROUP_IDS_ATTRIBUTE = "__gb_remoteGroupIds"


class _RemoteSavedGroupRequired(TypedDict):
    id: str
    attributeKey: str


class RemoteSavedGroup(_RemoteSavedGroupRequired, total=False):
    """A remote saved group from the payload, as a resolver sees it."""

    # For a derived group, the remote group whose ids it checks
    valuesFrom: str


# Looks up which of `groups` a user is in, from their `attributes` (only the ones
# remote groups use, keyed by `attributeKey`). Returns the group ids.
SavedGroupResolver = Callable[[Dict[str, Any], List[RemoteSavedGroup]], Awaitable[List[str]]]


def get_remote_saved_groups(saved_groups: Optional[Dict[str, Any]]) -> List[RemoteSavedGroup]:
    """The payload's remote saved groups. Only v2 payloads carry the metadata
    this reads.

    A new list of new entries on every call, so a caller (a resolver included)
    can't change what the next call sees. To check often, use
    RemoteSavedGroupsCache."""
    if not saved_groups:
        return []
    groups: List[RemoteSavedGroup] = []
    for group_id, entry in saved_groups.items():
        # A remote entry without an attribute gives a resolver nothing to look up.
        # Evaluation treats references to it as undecided.
        if isinstance(entry, dict) and entry.get("type") == "remote" and isinstance(entry.get("attributeKey"), str):
            group: RemoteSavedGroup = {"id": group_id, "attributeKey": entry["attributeKey"]}
            if entry.get("valuesFrom"):
                group["valuesFrom"] = entry["valuesFrom"]
            groups.append(group)
    return groups


class RemoteSavedGroupsCache:
    """Whether one client's saved groups have remote entries, scanned once per
    payload. A client swaps in a new saved groups dict per payload, so one slot
    per client is enough; keeping it per client means two clients with
    different payloads don't evict each other."""

    __slots__ = ("_source", "_has_remote")

    def __init__(self) -> None:
        self._source: Optional[Dict[str, Any]] = None
        self._has_remote = False

    def has_remote_groups(self, saved_groups: Optional[Dict[str, Any]]) -> bool:
        if not saved_groups:
            return False
        if saved_groups is not self._source:
            self._source = saved_groups
            self._has_remote = len(get_remote_saved_groups(saved_groups)) > 0
        return self._has_remote


def _get_path(attributes: Dict[str, Any], path: str) -> Any:
    from .core import getPath  # core imports this module

    return getPath(attributes, path)


async def resolve_remote_group_ids(
    attributes: Dict[str, Any],
    saved_groups: Optional[Dict[str, Any]],
    resolver: SavedGroupResolver,
) -> List[str]:
    """Asks `resolver` which of the payload's remote groups the user is in. Only
    the attributes those groups use are passed. Raises what the resolver raises.

    Returns [] when the payload lists no remote groups. Don't set that as the
    user's remote group ids: the payload may still check `__gb_remoteGroupIds`
    directly, and [] makes "not in group" rules match. Check
    `get_remote_saved_groups` first."""
    groups = get_remote_saved_groups(saved_groups)
    if not groups:
        return []
    used: Dict[str, Any] = {}
    for group in groups:
        # Keyed by `attributeKey`, which can be a dot-separated path
        value = _get_path(attributes, group["attributeKey"])
        if value is not None:
            used[group["attributeKey"]] = value
    return await resolver(used, groups)


def redis_resolver(redis: Any) -> SavedGroupResolver:
    """A resolver for remote groups loaded into Redis by GrowthBook's loader: one
    ``SMEMBERS gb:member:<attribute>:<value>`` per attribute value. Takes an
    async client with ``smembers``, such as ``redis.asyncio.Redis``.

    A derived group checks its own attribute's value in its source group's
    sets: `grp_vip:parent_id` looks up `gb:member:<grp_vip's attribute>:<parent_id>`
    and matches when `grp_vip` comes back."""

    async def resolve(attributes: Dict[str, Any], groups: List[RemoteSavedGroup]) -> List[str]:
        # For each group and value: the Redis key, and the id that means a match
        checks: List[Tuple[str, str, str]] = []
        for group in groups:
            source: Optional[RemoteSavedGroup] = group
            if "valuesFrom" in group:
                source = next((g for g in groups if g["id"] == group["valuesFrom"]), None)
            if source is None:
                continue
            for value in _to_lookup_values(attributes.get(group["attributeKey"])):
                checks.append((group["id"], f"gb:member:{source['attributeKey']}:{value}", source["id"]))
        keys = _unique([key for _, key, _ in checks])
        members = {key: {_decode(m) for m in await redis.smembers(key)} for key in keys}
        # Checking for the expected id, not any id in the set, also keeps keys
        # that collide when attributes or values contain ":" (`a:b` + `c` and
        # `a` + `b:c`) from mixing up groups.
        return _unique([group_id for group_id, key, match_id in checks if match_id in members[key]])

    return resolve


def is_missing_remote_group_ids(
    attributes: Optional[Dict[str, Any]],
    remote_group_ids: Optional[List[str]],
    saved_groups: Optional[Dict[str, Any]],
    cache: Optional[RemoteSavedGroupsCache] = None,
) -> bool:
    """Whether the payload has remote groups but the user has no remote group
    ids, so every rule using them will fail. The cheap checks come first, and
    `cache` spares rescanning the same payload."""
    if remote_group_ids is not None or isinstance((attributes or {}).get(REMOTE_GROUP_IDS_ATTRIBUTE), list):
        return False
    if cache is not None:
        return cache.has_remote_groups(saved_groups)
    return len(get_remote_saved_groups(saved_groups)) > 0


def with_remote_group_ids(attributes: Dict[str, Any], remote_group_ids: Optional[List[str]]) -> Dict[str, Any]:
    """The attributes conditions see: `attributes`, with the resolved remote group
    ids (when set) as `__gb_remoteGroupIds`, over any value the attributes set for
    it. Remote evaluation sends these too, so the server sees the same groups."""
    if remote_group_ids is None:
        return attributes
    return {**attributes, REMOTE_GROUP_IDS_ATTRIBUTE: list(remote_group_ids)}


def _to_lookup_values(value: Any) -> List[str]:
    """Each item of a list is looked up. Numbers use JS `String(value)`, the same
    form the loader stores them in."""
    from .core import _js_number_to_string  # core imports this module

    out: List[str] = []
    for v in value if isinstance(value, list) else [value]:
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            out.append(_js_number_to_string(v))
    return out


def _unique(values: List[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _decode(member: Any) -> str:
    return member.decode() if isinstance(member, bytes) else str(member)
