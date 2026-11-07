"""The model's chat format (engines/exl3_offload/qwen_chat.py): tool calls, sampling values and the template, and its
answers through the engine's splitter (glm53/stream_split.py) as the engine runs them. No model: the text is written
here."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines" / "exl3_offload"))
sys.path.insert(0, str(ROOT / "glm53-flash-offload" / "glm53"))
import qwen_chat as Q  # noqa: E402
import stream_split  # noqa: E402

WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                     "units": {"type": "string"}}}}}
SEARCH = {"type": "function", "function": {"name": "search", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}, "filters": {"type": "object"}}}}}
TOOLS = [WEATHER, SEARCH]
CALL = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nSan Francisco\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")


def args(call):
    return json.loads(call["function"]["arguments"])


class ToolCallTests(unittest.TestCase):
    def test_a_call_becomes_an_openai_tool_call(self):
        text, calls = Q.parse_tool_calls("Let me look that up.\n" + CALL, TOOLS)
        self.assertEqual(text, "Let me look that up.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["type"], "function")
        self.assertTrue(calls[0]["id"].startswith("call_"))
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(args(calls[0]), {"city": "San Francisco", "days": 3})   # typed by the tool's schema

    def test_string_parameters_stay_text_and_keep_their_inner_lines(self):
        code = "def f():\n    return 1\n"
        _, calls = Q.parse_tool_calls(f"<tool_call>\n<function=search>\n<parameter=query>\n{code}\n</parameter>\n"
                                      f"<parameter=filters>\n{{\"lang\": \"py\"}}\n</parameter>\n</function>\n"
                                      f"</tool_call>", TOOLS)
        self.assertEqual(args(calls[0]), {"query": code, "filters": {"lang": "py"}})
        _, calls = Q.parse_tool_calls("<tool_call><function=get_weather><parameter=units>42</parameter>"
                                      "</function></tool_call>", TOOLS)
        self.assertEqual(args(calls[0]), {"units": "42"})                      # a string, though it parses as JSON

    def test_several_calls_and_text_between_them(self):
        second = CALL.replace("San Francisco", "Oslo")
        text, calls = Q.parse_tool_calls(f"First.\n{CALL}\nThen.\n{second}", TOOLS)
        self.assertEqual([args(c)["city"] for c in calls], ["San Francisco", "Oslo"])
        self.assertEqual(text, "First.\n\nThen.")
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])

    def test_a_call_cut_off_by_the_end_of_the_answer_still_counts(self):
        _, calls = Q.parse_tool_calls("<tool_call>\n<function=get_weather>\n<parameter=city>\nRome", TOOLS)
        self.assertEqual(args(calls[0]), {"city": "Rome"})

    def test_unknown_parameters_are_json_where_they_parse(self):
        _, calls = Q.parse_tool_calls("<tool_call><function=other><parameter=n>7</parameter>"
                                      "<parameter=s>hi</parameter></function></tool_call>", TOOLS)
        self.assertEqual(args(calls[0]), {"n": 7, "s": "hi"})

    def test_without_tools_or_a_call_the_text_is_left_alone(self):
        self.assertEqual(Q.parse_tool_calls(CALL, None), (CALL, []))
        self.assertEqual(Q.parse_tool_calls("No call here.", TOOLS), ("No call here.", []))
        broken = "<tool_call>not a function</tool_call>"
        self.assertEqual(Q.parse_tool_calls(broken, TOOLS), (broken, []))     # nothing parsed: shown as written


class StreamTests(unittest.TestCase):
    """Qwen3.8's answers through the engine's splitter, its tool calls parsed at the end (serve.py's turn_events)."""

    def run_stream(self, pieces, thinking=False, tools=TOOLS):
        s = stream_split.StreamSplit(thinking, bool(tools))
        events = [e for p in pieces for e in s.feed(p)] + s.finish()
        if tools and s.tool_text:
            rest, calls = Q.parse_tool_calls(s.tool_text, tools)
            events += [("content", rest)] if rest else []
            events += [("tool_calls", calls)] if calls else []
        joined = {k: "".join(v for kind, v in events if kind == k) for k in ("reasoning", "content")}
        calls = [c for kind, v in events if kind == "tool_calls" for c in v]
        return events, joined, calls

    def test_prose_streams_while_tools_are_offered(self):
        words = "Heat pumps move heat with a refrigerant loop. ".split(" ")
        events, joined, calls = self.run_stream([w + " " for w in words])
        content = [v for kind, v in events if kind == "content"]
        self.assertGreater(len(content), 5)                                     # piece by piece, not at the end
        self.assertEqual(joined["content"], "".join(w + " " for w in words))
        self.assertEqual(calls, [])

    def test_a_call_never_reaches_the_content_wherever_the_pieces_break(self):
        text = "Checking the forecast.\n" + CALL + "\n"
        for cut in range(len(text) + 1):
            head, rest = text[:cut], text[cut:]
            for pieces in ((head, rest), (head, rest[:3], rest[3:])):
                events, joined, calls = self.run_stream(pieces)
                with self.subTest(cut=cut):
                    self.assertEqual(joined["content"], "Checking the forecast.\n")
                    self.assertNotIn("<tool", joined["content"])
                    self.assertEqual([args(c) for c in calls], [{"city": "San Francisco", "days": 3}])
        _, joined, calls = self.run_stream(list(text))                          # one character at a time
        self.assertEqual((joined["content"], len(calls)), ("Checking the forecast.\n", 1))

    def test_a_tag_that_turns_out_to_be_text_is_sent_after_all(self):
        _, joined, calls = self.run_stream(["Use the <tool", "box> command, x < 2."])
        self.assertEqual(joined["content"], "Use the <toolbox> command, x < 2.")
        _, joined, _ = self.run_stream(["Ends with <tool_"])                     # held to the end, then sent
        self.assertEqual(joined["content"], "Ends with <tool_")

    def test_reasoning_ends_at_the_closing_tag_wherever_it_breaks(self):
        text = "The user wants a forecast.</think>\n\nIt will rain."
        for cut in range(len(text) + 1):
            _, joined, _ = self.run_stream((text[:cut], text[cut:]), thinking=True, tools=None)
            with self.subTest(cut=cut):
                self.assertEqual(joined["reasoning"], "The user wants a forecast.")
                self.assertEqual(joined["content"], "\n\nIt will rain.")

    def test_reasoning_then_a_call(self):
        _, joined, calls = self.run_stream(["<think>Need data.", "</think>\n", CALL[:20], CALL[20:]], thinking=True)
        self.assertEqual(joined["reasoning"], "Need data.")
        self.assertEqual(joined["content"], "\n")
        self.assertEqual(len(calls), 1)

    def test_without_tools_markup_is_plain_text(self):
        _, joined, calls = self.run_stream(["a ", CALL], tools=None)
        self.assertEqual((joined["content"], calls), ("a " + CALL, []))

    def test_a_call_that_does_not_parse_is_shown_as_written(self):
        _, joined, calls = self.run_stream(["Here: ", "<tool_call>not a function</tool_call>"])
        self.assertEqual((joined["content"], calls), ("Here: <tool_call>not a function</tool_call>", []))


class WholeAnswerTests(unittest.TestCase):
    """A finished answer, as serve.py's turn_answer splits it."""

    @staticmethod
    def whole(text, thinking, tools):
        reasoning, content = stream_split.split_answer(text, thinking)
        return (reasoning, *Q.parse_tool_calls(content, tools))

    def test_reasoning_content_and_calls(self):
        reasoning, content, calls = self.whole("Think.</think>\n\nSure.\n" + CALL, True, TOOLS)
        self.assertEqual((reasoning, content, len(calls)), ("Think.", "Sure.", 1))

    def test_a_think_block_never_closed_is_all_reasoning(self):
        self.assertEqual(self.whole("<think>Still thinking", True, None), ("Still thinking", "", []))

    def test_thinking_off(self):
        self.assertEqual(self.whole("  Plain.  ", False, None), (None, "Plain.", []))


class SamplingTests(unittest.TestCase):
    def test_the_models_values_fill_what_the_request_leaves_out(self):
        self.assertEqual(Q.with_sampling({}), {**Q.SAMPLING[True]})
        off = Q.with_sampling({"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0, "top_p": None})
        self.assertEqual((off["temperature"], off["top_p"], off["top_k"]), (0, 0.8, 20))
        self.assertEqual(Q.with_sampling({"top_k": 5})["top_k"], 5)


@unittest.skipUnless(__import__("importlib").util.find_spec("jinja2"), "needs jinja2 (the engine's environment)")
class TemplateTests(unittest.TestCase):
    SOURCE = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}"
              "{% if tools %}TOOLS {{ tools | tojson }}\n{% endif %}"
              "{% if add_generation_prompt %}<|assistant|>{% if enable_thinking %}<think>{% else %}<think></think>"
              "{% endif %}{% endif %}{{ reasoning_effort | default('') }}")

    def test_it_renders_the_conversation_and_the_options(self):
        t = Q.ChatTemplate(self.SOURCE)
        out = t.render([{"role": "user", "content": "Hi"}], [{"name": "x"}], False, {"reasoning_effort": "low",
                                                                                  "enable_thinking": True})
        self.assertEqual(out, '<|user|>Hi\nTOOLS [{"name": "x"}]\n<|assistant|><think></think>low')

    def test_the_sandbox_refuses_python(self):
        from jinja2.exceptions import SecurityError
        t = Q.ChatTemplate("{{ ''.__class__.__mro__ }}")
        with self.assertRaises(SecurityError):
            t.render([], thinking=True)

    def test_raise_exception_is_a_template_error(self):
        with self.assertRaises(Q.TemplateError):
            Q.ChatTemplate("{{ raise_exception('bad role') }}").render([])


if __name__ == "__main__":
    unittest.main()
