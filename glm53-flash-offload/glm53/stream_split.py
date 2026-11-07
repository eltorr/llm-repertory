"""Split a chat answer into reasoning, content and tool calls while it streams (serve.py's chat endpoint).

Tag-aware hold-back: text that could still turn into a tag is not sent until later text shows it does not.
  - reasoning: everything up to the closing think tag (opening tags removed); a partial closing tag is held back
  - content: sent as it comes; with tools, a partial tool-call tag at the end of the text is held back, and from the
    first complete tool-call tag on everything is kept (tool_text) for the caller's parser at the end of the answer,
    so a tool call never reaches the client as plain text
  - an answer whose prompt did not open a think block may open one itself as its first output
Pure Python (no torch); the tags are parameters (serve.py passes its THINK_OPEN, THINK_CLOSE, TOOL_OPEN).

    s = StreamSplit(in_think=prompt_opens_think(prompt), tools=bool(tools))
    for kind, text in s.feed(new_text): ...     kind: "reasoning" | "content"
    for kind, text in s.finish(): ...           end of the answer: whatever was held back
    s.tool_text                                 from the first tool-call tag on ("" if none)
"""


def prompt_opens_think(prompt, think_open="<think>", think_close="</think>"):
    """True when the prompt ends inside a think block (its last opening tag is not closed), i.e. the answer starts as
    reasoning. GLM-5.3 and Qwen3.8 templates open one at the end of the generation prompt when thinking is on and
    write an empty, closed one when it is off."""
    i = prompt.rfind(think_open)
    return i >= 0 and prompt.find(think_close, i + len(think_open)) < 0


def held(text, tags):
    """Length of the longest end of `text` that is the start of one of `tags` (text that could still become a tag)."""
    best = 0
    for t in tags:
        for k in range(min(len(t) - 1, len(text)), best, -1):
            if t.startswith(text[-k:]):
                best = k
                break
    return best


def split_answer(text, in_think, think_open="<think>", think_close="</think>"):
    """The whole-answer form (non-streamed responses): (reasoning or None, content). A think block that never closes
    (the answer hit its token limit) is all reasoning."""
    if not in_think:
        s = text.lstrip()
        if not s.startswith(think_open):
            return None, text.strip()
        text = s[len(think_open):]
    r, sep, c = text.partition(think_close)
    if not sep:
        return text.replace(think_open, "").strip(), ""
    return r.replace(think_open, "").strip(), c.strip()


class StreamSplit:
    def __init__(self, in_think, tools, think_open="<think>", think_close="</think>", tool_open="<tool_call>"):
        self.reasoning = bool(in_think)
        self.tools = bool(tools)
        self.think_open, self.think_close, self.tool_open = think_open, think_close, tool_open
        self.buf = ""            # received, not sent yet
        self.tool_text = ""      # from the first tool-call tag on
        self.started = False     # content has begun: leading blank lines dropped, no think block can open any more

    def feed(self, text):
        out = []
        self.buf += text
        while self.buf:
            if self.tool_text:
                self.tool_text += self.buf
                self.buf = ""
            elif self.reasoning:
                i = self.buf.find(self.think_close)
                if i >= 0:
                    self._emit(out, "reasoning", self.buf[:i])
                    self.buf = self.buf[i + len(self.think_close):]
                    self.reasoning = False
                    continue
                k = held(self.buf, (self.think_close, self.think_open))
                self._emit(out, "reasoning", self.buf[:len(self.buf) - k])
                self.buf = self.buf[len(self.buf) - k:]
                break
            else:
                if not self.started:
                    s = self.buf.lstrip()
                    if not s or (len(s) < len(self.think_open) and self.think_open.startswith(s)):
                        break                       # blank so far, or maybe the start of a think block
                    if s.startswith(self.think_open):
                        self.buf = s[len(self.think_open):]
                        self.reasoning = True
                        continue
                    self.buf, self.started = s, True
                if self.tools:
                    i = self.buf.find(self.tool_open)
                    if i >= 0:
                        self._emit(out, "content", self.buf[:i])
                        self.tool_text, self.buf = self.buf[i:], ""
                        break
                k = held(self.buf, (self.tool_open,)) if self.tools else 0
                self._emit(out, "content", self.buf[:len(self.buf) - k])
                self.buf = self.buf[len(self.buf) - k:]
                break
        return out

    def finish(self):
        """End of the answer: what was held back was no tag after all."""
        out = []
        if self.buf and not self.tool_text:
            if self.reasoning:
                self._emit(out, "reasoning", self.buf)
            else:
                self._emit(out, "content", self.buf if self.started else self.buf.lstrip())
        self.buf = ""
        return out

    def _emit(self, out, kind, text):
        if kind == "reasoning":
            text = text.replace(self.think_open, "")
        if text:
            out.append((kind, text))
