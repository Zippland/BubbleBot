"""Parse native WeChat membership notices (message type 10000)."""

import re

# Quoted names may themselves contain separators such as 、 or 和. Match whole
# names before splitting the member list so we never welcome the inviter.
_NAME = r'(?:"[^"\r\n]+"|“[^”\r\n]+”|[^"“”\s、，,]+)'
_MEMBERS = rf'{_NAME}(?:\s*(?:、|，|,|和)\s*{_NAME})*'
_JOIN_PATTERNS = tuple(re.compile(pattern) for pattern in (
    rf'{_NAME}\s*邀请\s*(?P<members>{_MEMBERS})\s*加入了?群聊[。.]?',
    rf'(?P<members>{_MEMBERS})\s*通过扫描(?:{_NAME}分享的)?二维码加入了?群聊[。.]?',
    rf'(?P<members>{_MEMBERS})\s*通过{_NAME}的邀请加入了?群聊[。.]?',
    rf'(?P<members>{_MEMBERS})\s*加入了?群聊[。.]?',
))
_MEMBER_NAME = re.compile(rf'(?P<name>{_NAME})(?:\s*(?:、|，|,|和)\s*|$)')


def parse_group_join_members(content: str | None) -> list[str]:
    """Return newcomers from a native join notice, excluding the bot (你).

    Callers must check the native message type; ordinary user text is not an
    event even if it contains the same words. Unknown system notices stay out.
    """
    if not content:
        return []
    for pattern in _JOIN_PATTERNS:
        match = pattern.fullmatch(content.strip())
        if match is None:
            continue
        names = []
        for member in _MEMBER_NAME.finditer(match['members']):
            raw_name = member['name']
            # Only an unquoted 你 means the logged-in bot. A user may really
            # have the nickname "你".
            if raw_name == '你':
                continue
            name = raw_name[1:-1] if raw_name[0] in '"“' else raw_name
            if name.strip() and name not in names:
                names.append(name)
        return names
    return []
