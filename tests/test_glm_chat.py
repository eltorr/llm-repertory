"""GLM-5.3-Flash's chat format (engines/exl3_offload/glm_chat.py): tool calls, sampling values, reasoning levels, the
thinking switch and the template; and which format the model server picks for a model folder, with its served name
and stop tokens (server.chat_format, served_name, stop_tokens). No model: the text is written here. The template tests
need jinja2 (the engine's environment); GLM_MODEL_DIR=/path/to/model-glm-3.05 also renders the model's own template."""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines" / "exl3_offload"))
sys.path.insert(0, str(ROOT / "glm53-flash-offload" / "glm53"))
import glm_chat as G  # noqa: E402
import openai_compat  # noqa: E402
import qwen_chat as Q  # noqa: E402
import server  # noqa: E402
import stream_split  # noqa: E402

WEATHER = {"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"},
                                     "units": {"type": "string"}}}}}
SEARCH = {"type": "function", "function": {"name": "search", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}, "filters": {"type": "object"}}}}}
TOOLS = [WEATHER, SEARCH]
CALL = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Lisbon</arg_value><arg_key>days</arg_key>" \
       "<arg_value>3</arg_value></tool_call>"
HAVE_JINJA = __import__("importlib").util.find_spec("jinja2") is not None


def args(call):
    return json.loads(call["function"]["arguments"])


class ToolCallTests(unittest.TestCase):
    def test_a_call_becomes_an_openai_tool_call(self):
        text, calls = G.parse_tool_calls("Let me check." + CALL, TOOLS)
        self.assertEqual(text, "Let me check.")
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["type"], calls[0]["function"]["name"]), ("function", "get_weather"))
        self.assertTrue(calls[0]["id"].startswith("call_"))
        self.assertEqual(args(calls[0]), {"city": "Lisbon", "days": 3})          # a string stays text, 3 is JSON

    def test_string_parameters_stay_text(self):
        call = "<tool_call>search<arg_key>query</arg_key><arg_value>42</arg_value><arg_key>filters</arg_key>" \
               '<arg_value>{"lang": "pt"}</arg_value></tool_call>'
        _, calls = G.parse_tool_calls(call, TOOLS)
        self.assertEqual(args(calls[0]), {"query": "42", "filters": {"lang": "pt"}})

    def test_several_calls_and_a_call_without_arguments(self):
        text = CALL + "\n" + "<tool_call>search</tool_call>"
        rest, calls = G.parse_tool_calls(text, TOOLS)
        self.assertEqual([c["function"]["name"] for c in calls], ["get_weather", "search"])
        self.assertEqual((args(calls[1]), rest), ({}, ""))
        self.assertEqual(len({c["id"] for c in calls}), 2)

    def test_a_call_cut_off_by_the_end_of_the_answer_still_counts(self):
        _, calls = G.parse_tool_calls("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Lisb", TOOLS)
        self.assertEqual(args(calls[0]), {"city": "Lisb"})

    def test_unknown_functions_take_json_where_it_parses(self):
        _, calls = G.parse_tool_calls("<tool_call>other<arg_key>n</arg_key><arg_value>7</arg_value>"
                                      "<arg_key>s</arg_key><arg_value>seven</arg_value></tool_call>", TOOLS)
        self.assertEqual((calls[0]["function"]["name"], args(calls[0])), ("other", {"n": 7, "s": "seven"}))

    def test_without_tools_or_a_call_that_parses_the_text_is_left_alone(self):
        self.assertEqual(G.parse_tool_calls(CALL, None), (CALL, []))
        self.assertEqual(G.parse_tool_calls("No call here.", TOOLS), ("No call here.", []))
        qwen = "<tool_call>\n<function=get_weather>\n<parameter=city>\nLisbon\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(G.parse_tool_calls(qwen, TOOLS), (qwen, []))            # Qwen's format isn't GLM's


class StreamTests(unittest.TestCase):
    """GLM's answer through the engine's splitter, at its tags: reasoning, then prose, then the call held back."""

    def test_reasoning_prose_and_a_call(self):
        split = stream_split.StreamSplit(True, True, G.THINK_OPEN, G.THINK_CLOSE, G.TOOL_OPEN)
        answer = "Need the weather.</think>I'll check." + CALL
        events = [ev for i in range(0, len(answer), 5) for ev in split.feed(answer[i:i + 5])] + list(split.finish())
        self.assertEqual("".join(t for k, t in events if k == "reasoning"), "Need the weather.")
        self.assertEqual("".join(t for k, t in events if k == "content"), "I'll check.")
        self.assertEqual(G.parse_tool_calls(split.tool_text, TOOLS)[1][0]["function"]["name"], "get_weather")

    def test_thinking_off_is_all_content(self):
        reasoning, content = stream_split.split_answer("Plain.", False, G.THINK_OPEN, G.THINK_CLOSE)
        self.assertEqual((reasoning, content), (None, "Plain."))


class SamplingAndEffortTests(unittest.TestCase):
    def test_generation_configs_values_fill_what_the_request_leaves_out(self):
        self.assertEqual(G.with_sampling({}), {"temperature": 1.0, "top_p": 0.95})
        off = G.with_sampling({"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0, "top_p": None})
        self.assertEqual((off["temperature"], off["top_p"]), (0, 0.95))
        self.assertNotIn("top_k", off)

    def test_openai_levels_become_glms(self):
        def level(effort):
            return openai_compat.normalize_chat_request({"reasoning_effort": effort}, G.EFFORTS).get("reasoning_effort")
        self.assertEqual([level(e) for e in ("minimal", "low", "medium", "high", "xhigh", "max")],
                         ["low", "low", "high", "high", "max", "max"])
        off = openai_compat.normalize_chat_request({"reasoning": {"effort": "none"}}, G.EFFORTS)
        self.assertEqual(off["chat_template_kwargs"], {"enable_thinking": False})
        # Qwen's levels are the engine's default
        self.assertEqual(Q.EFFORTS, openai_compat.EFFORT_MAP)
        self.assertEqual(openai_compat.normalize_chat_request({"reasoning_effort": "high"})["reasoning_effort"],
                         "xhigh")

    def test_both_formats_have_the_same_names(self):
        for name in ("NAME", "ChatTemplate", "parse_tool_calls", "with_sampling", "EFFORTS", "END_OF_TURN",
                     "THINK_OPEN", "THINK_CLOSE", "TOOL_OPEN", "TemplateError"):
            self.assertTrue(hasattr(G, name) and hasattr(Q, name), name)
        self.assertEqual(G.END_OF_TURN, ("<|endoftext|>", "<|user|>", "<|observation|>"))


# GLM-5.3's generation prompt as its template writes it: always the open think block, no enable_thinking
GLM_LIKE = ("[gMASK]<sop><|system|>Reasoning Effort: {{ reasoning_effort if reasoning_effort in ['low', 'high'] "
            "else 'max' }}{% if tools %}<|system|>TOOLS {{ tools | tojson(ensure_ascii=False) }}{% endif %}"
            "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|><think>{% endif %}")


@unittest.skipUnless(HAVE_JINJA, "needs jinja2 (the engine's environment)")
class TemplateTests(unittest.TestCase):
    def test_thinking_on_leaves_the_think_block_open_off_writes_it_empty(self):
        t = G.ChatTemplate(GLM_LIKE)
        on = t.render([{"role": "user", "content": "Hi"}], None, True, {})
        off = t.render([{"role": "user", "content": "Hi"}], None, False, {"reasoning_effort": "low"})
        self.assertEqual(on, "[gMASK]<sop><|system|>Reasoning Effort: max<|user|>Hi<|assistant|><think>")
        self.assertEqual(off, "[gMASK]<sop><|system|>Reasoning Effort: low<|user|>Hi<|assistant|><think></think>")
        self.assertTrue(stream_split.prompt_opens_think(on, G.THINK_OPEN, G.THINK_CLOSE))
        self.assertFalse(stream_split.prompt_opens_think(off, G.THINK_OPEN, G.THINK_CLOSE))

    def test_tools_are_written_as_json(self):
        out = G.ChatTemplate(GLM_LIKE).render([{"role": "user", "content": "Où?"}], [WEATHER], False)
        self.assertIn('TOOLS [{"type": "function"', out)
        self.assertIn("Où?", out)

    def test_the_sandbox_refuses_python(self):
        from jinja2.exceptions import SecurityError
        with self.assertRaises(SecurityError):
            G.ChatTemplate("{{ ''.__class__.__mro__ }}").render([])

    @unittest.skipUnless(os.environ.get("GLM_MODEL_DIR"), "set GLM_MODEL_DIR to the downloaded model folder")
    def test_the_models_own_template(self):
        t = G.ChatTemplate.from_model(os.environ["GLM_MODEL_DIR"])
        q = [{"role": "user", "content": "What's the weather in Lisbon right now?"}]
        self.assertTrue(t.render(q, None, True).endswith("<|assistant|><think>"))
        self.assertTrue(t.render(q, None, False).endswith("<|assistant|><think></think>"))
        self.assertIn("Reasoning Effort: High", t.render(q, None, True, {"reasoning_effort": "high"}))
        with_tools = t.render(q, [WEATHER], False)
        self.assertIn('"name": "get_weather"', with_tools)
        self.assertIn("<arg_key>", with_tools)                                   # the call format it asks for


class ServerChoiceTests(unittest.TestCase):
    """The model server picks the format by config.json's model_type, and the served name by the catalog."""

    def folder(self, cfg):
        d = Path(tempfile.mkdtemp(prefix="repertory-model-"))
        self.addCleanup(__import__("shutil").rmtree, d, True)
        if cfg is not None:
            (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        return d

    def test_the_format_follows_the_model_type(self):
        glm = self.folder({"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next"})
        qwen = self.folder({"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp"})
        self.assertIs(server.chat_format(glm), G)
        self.assertIs(server.chat_format(qwen), Q)
        for bad in (self.folder({"model_type": "llama"}), self.folder(None), self.folder([1, 2])):
            with self.assertRaises(ValueError):
                server.chat_format(bad)

    def test_the_served_name_is_the_catalogs_for_the_folder(self):
        self.assertEqual(server.served_name(server.ROOT / "model-glm-3.05"), "glm-5.3-flash")
        self.assertEqual(server.served_name(server.ROOT / "store" / ".." / "model-4.05"), "qwen3.8-flash-next")
        self.assertEqual(server.served_name(server.ROOT / "my-model"), "my-model")      # not in the catalog
        self.assertIsNone(server.parse_args(["--model", "m", "--store", "s", "--ram-gb", "4"]).served_name)

    def test_glms_stop_tokens(self):
        ids = {"<|endoftext|>": 154820, "<|user|>": 154827, "<|observation|>": 154829}

        def single_id(name):
            return ids[name]                                  # KeyError for a token GLM doesn't have
        tok = types.SimpleNamespace(eos_token_id=154820, single_id=single_id)
        d = self.folder({"model_type": "glm5_next"})
        (d / "generation_config.json").write_text(json.dumps({"eos_token_id": [154820, 154827, 154829]}))
        cfg = types.SimpleNamespace(eos_token_id_list=[154820])
        self.assertEqual(server.stop_tokens(cfg, tok, d, G.END_OF_TURN), [154820, 154827, 154829])
        self.assertEqual(server.stop_tokens(cfg, tok, d, Q.END_OF_TURN), [154820, 154827, 154829])   # no im_end


if __name__ == "__main__":
    unittest.main()
