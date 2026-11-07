"""glm53/stream_split.py: reasoning / content / tool-call split of a streamed answer, with tag hold-back. Every answer
is fed whole, one character at a time and in every 2- and 3-piece split; whatever the chunking, the client must see the
same reasoning and content, and no piece of a tool call may ever arrive as content. Pure Python (no torch)."""
import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "glm53"))
import stream_split as S  # noqa: E402

QWEN_ON = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
QWEN_OFF = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
GLM_ON = "[gMASK]<sop><|user|>hi<|assistant|><think>"
GLM_OFF = "[gMASK]<sop><|user|>hi<|assistant|><think></think>"
CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"


def chunkings(text):
    yield [text]
    yield list(text)
    for i in range(len(text) + 1):
        yield [text[:i], text[i:]]
    step = max(1, len(text) // 12)
    for i, j in itertools.combinations(range(0, len(text) + 1, step), 2):
        yield [text[:i], text[i:j], text[j:]]


def run(pieces, in_think, tools):
    s = S.StreamSplit(in_think, tools)
    out = []
    for p in pieces:
        out += s.feed(p)
    out += s.finish()
    return out, s.tool_text


def joined(out, kind):
    return "".join(t for k, t in out if k == kind)


class PromptState(unittest.TestCase):
    def test_templates(self):
        self.assertTrue(S.prompt_opens_think(QWEN_ON))
        self.assertFalse(S.prompt_opens_think(QWEN_OFF))
        self.assertTrue(S.prompt_opens_think(GLM_ON))
        self.assertFalse(S.prompt_opens_think(GLM_OFF))
        self.assertFalse(S.prompt_opens_think("no tags at all"))
        # an earlier, closed think block in the history does not count
        hist = "<|im_start|>assistant\n<think>\nold\n</think>\n\nanswer<|im_end|>\n" + QWEN_ON
        self.assertTrue(S.prompt_opens_think(hist))
        self.assertFalse(S.prompt_opens_think(hist.replace(QWEN_ON, QWEN_OFF)))

    def test_held(self):
        self.assertEqual(S.held("abc <tool_c", ("<tool_call>",)), 7)
        self.assertEqual(S.held("abc <tool_call>", ("<tool_call>",)), 0)   # complete tags are found, not held
        self.assertEqual(S.held("a < b", ("<tool_call>",)), 0)
        self.assertEqual(S.held("x</thi", ("</think>", "<think>")), 5)
        self.assertEqual(S.held("x<", ("</think>", "<think>")), 1)


class Split(unittest.TestCase):
    def check(self, answer, in_think, tools, reasoning, content, tool_text):
        for pieces in chunkings(answer):
            out, tt = run(pieces, in_think, tools)
            self.assertEqual(joined(out, "reasoning"), reasoning, pieces)
            self.assertEqual(joined(out, "content"), content, pieces)
            self.assertEqual(tt, tool_text, pieces)
            for k, t in out:
                self.assertTrue(t, "empty piece")
            if tools:   # a tool-call tag never reaches the client as content, whole or split over pieces
                self.assertNotIn("<tool_call>", joined(out, "content"), pieces)

    def test_reasoning_then_content(self):
        self.check("Let me think.\n</think>\n\nHello!", True, False, "Let me think.\n", "Hello!", "")

    def test_reasoning_strips_open_tag(self):
        self.check("<think>abc</think>def", True, False, "abc", "def", "")

    def test_no_thinking(self):
        self.check("\n\nPlain answer < with > signs", False, False, "", "Plain answer < with > signs", "")

    def test_answer_opens_its_own_think_block(self):
        self.check("<think>hmm</think>\nOK", False, False, "hmm", "OK", "")

    def test_think_tag_later_in_content_is_text(self):
        self.check("Use <think> tags.", False, False, "", "Use <think> tags.", "")

    def test_tool_call_after_reasoning(self):
        ans = "I should call it.\n</think>\n\n" + CALL
        self.check(ans, True, True, "I should call it.\n", "", CALL)

    def test_content_before_tool_call(self):
        ans = "Checking the weather.\n\n" + CALL + "\n" + CALL
        self.check(ans, False, True, "", "Checking the weather.\n\n", CALL + "\n" + CALL)

    def test_tag_lookalike_is_released(self):
        # "<tool" that never becomes a tool call is content, at the end at the latest
        self.check("a <tool b <tool_c", False, True, "", "a <tool b <tool_c", "")
        self.check("x <tool_call", False, True, "", "x <tool_call", "")

    def test_without_tools_tags_stream_as_content(self):
        self.check("Here: " + CALL, False, False, "", "Here: " + CALL, "")

    def test_unclosed_reasoning(self):
        self.check("still thinking </thi", True, True, "still thinking </thi", "", "")

    def test_tool_tag_inside_reasoning_is_reasoning(self):
        ans = "maybe <tool_call> later</think>done"
        self.check(ans, True, True, "maybe <tool_call> later", "done", "")

    def test_content_is_streamed_early(self):
        s = S.StreamSplit(False, True)
        self.assertEqual(s.feed("Hello wor"), [("content", "Hello wor")])
        self.assertEqual(s.feed("ld <tool"), [("content", "ld ")])      # "<tool" held
        self.assertEqual(s.feed("_call>\n<function=f>"), [])              # a tool call: held from here on
        self.assertEqual(s.finish(), [])
        self.assertEqual(s.tool_text, "<tool_call>\n<function=f>")


class WholeAnswer(unittest.TestCase):
    def test_split_answer(self):
        self.assertEqual(S.split_answer("r\n</think>\n\nc", True), ("r", "c"))
        self.assertEqual(S.split_answer("never closed", True), ("never closed", ""))
        self.assertEqual(S.split_answer("  plain  ", False), (None, "plain"))
        self.assertEqual(S.split_answer("<think>r</think>c", False), ("r", "c"))
        self.assertEqual(S.split_answer("<think>r</think>c", True), ("r", "c"))

    def test_stream_matches_whole_answer(self):
        for ans, in_think in (("r\n</think>\n\nc d", True), ("plain\n", False), ("<think>x</think>y", False)):
            out, _ = run(list(ans), in_think, False)
            r, c = S.split_answer(ans, in_think)
            self.assertEqual(joined(out, "content").strip(), c)
            self.assertEqual(joined(out, "reasoning").strip(), r or "")


if __name__ == "__main__":
    unittest.main()
