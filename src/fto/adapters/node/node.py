from typing import Any, List

from aegis_mas.aegis_core import AgentContext


def block_text(block: Any) -> str:
    """Text carried by one content block of a multimodal message.

    Returns '' for anything that is not a text block -- an attachment, an
    image, a payload we don't recognise -- so callers can tell apart "no text
    here" from real content. Handles both object-shaped blocks (an attribute
    pair of ``type``/``text``) and the dict form providers use on the wire.
    """
    if isinstance(block, dict):
        if block.get('type', 'text') != 'text':
            return ''
        return block.get('text') or ''
    block_type = getattr(block, 'type', None)
    if block_type is not None and block_type != 'text':
        return ''
    return getattr(block, 'text', None) or ''


def content_text(content: Any) -> str:
    """Text of a message's content, whichever shape it comes in.

    A plain string is itself; a block list contributes only its text blocks,
    joined; anything else has no text to offer.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    return '\n\n'.join(text for text in map(block_text, content) if text)


def replace_content_text(content: Any, text: str, make_text_block) -> Any:
    """Same content with its text replaced by ``text``, or None if it has none.

    A plain string is replaced outright. In a block list the non-text blocks
    (attachments, images) keep their place and ``text`` takes over the first
    text slot, the remaining text blocks dropping out -- ``text`` already
    stands for all of them.
    """
    if isinstance(content, str):
        return text
    if not isinstance(content, list):
        return None

    rebuilt = []
    replaced = False
    for block in content:
        if not block_text(block):
            rebuilt.append(block)
            continue
        if replaced:
            continue
        rebuilt.append(make_text_block(block, text))
        replaced = True
    return rebuilt if replaced else None


class NodeAdapter:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @property
    def id(self) -> str:
        pass

    @property
    def input(self) -> Any:
        pass

    @property
    def last_message(self) -> Any:
        pass

    @property
    def is_agent(self) -> bool:
        pass

    def set_input(self, value: Any) -> None:
        pass

    def to_aegis_context(self) -> AgentContext:
        pass

    def append_to_last_message(self, text: str) -> None:
        pass

    def overwrite_last_message(self, text: str) -> None:
        pass

    def context_as_list(self, context: Any = None) -> List[str]:
        """Plain-text view of a node context, one entry per message.

        ``context`` defaults to the node's current input; pass a snapshot to
        read that instead. Entries holding nothing a compressor may rewrite --
        empty content, attachment-only content, tool-protocol messages whose
        pairing with a call must stay intact -- come back as '' so callers
        leave them alone.
        """
        pass

    def context_from_list(self, texts: List[str], context: Any = None) -> Any:
        """Rebuild a node context with each message's text replaced by ``texts``.

        ``texts`` lines up with ``context`` entry for entry, and '' keeps that
        message exactly as it was. Only the text inside a message changes:
        roles, sources, keep flags and attachments are carried over untouched,
        which is what keeps the rebuilt list valid to send to a provider.
        """
        pass
