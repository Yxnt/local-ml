import unittest
import json
from backends.tool_parser import parse_mcp_tool_calls


class ParseMcpToolCallsTests(unittest.TestCase):
    def test_single_tool_call_with_one_param(self):
        text = "<function=get_weather>\n  <parameter=city>Beijing</parameter>\n</function>"
        result = parse_mcp_tool_calls(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "get_weather")
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args, {"city": "Beijing"})

    def test_single_tool_call_with_multiple_params(self):
        text = (
            "<function=search>\n"
            "  <parameter=query>python tutorial</parameter>\n"
            "  <parameter=max_results>10</parameter>\n"
            "  <parameter=language>en</parameter>\n"
            "</function>"
        )
        result = parse_mcp_tool_calls(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "search")
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args["query"], "python tutorial")
        self.assertEqual(args["max_results"], "10")
        self.assertEqual(args["language"], "en")

    def test_multiple_tool_calls(self):
        text = (
            "<function=get_weather>\n"
            "  <parameter=city>Beijing</parameter>\n"
            "</function>\n"
            "<function=get_time>\n"
            "  <parameter=timezone>Asia/Shanghai</parameter>\n"
            "</function>"
        )
        result = parse_mcp_tool_calls(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["function"]["name"], "get_weather")
        self.assertEqual(result[1]["function"]["name"], "get_time")

    def test_tool_call_with_surrounding_text(self):
        text = (
            "I will help you check the weather.\n"
            "<function=get_weather>\n"
            "  <parameter=city>Shanghai</parameter>\n"
            "</function>\n"
            "Let me fetch that for you."
        )
        result = parse_mcp_tool_calls(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "get_weather")

    def test_no_tool_calls_returns_empty_list(self):
        text = "This is a regular message with no tool calls."
        result = parse_mcp_tool_calls(text)
        self.assertEqual(result, [])

    def test_empty_string_returns_empty_list(self):
        result = parse_mcp_tool_calls("")
        self.assertEqual(result, [])

    def test_tool_call_with_empty_params(self):
        text = "<function=get_status>\n</function>"
        result = parse_mcp_tool_calls(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "get_status")
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args, {})

    def test_tool_call_with_multiline_param_value(self):
        text = (
            "<function=write_file>\n"
            "  <parameter=path>/tmp/test.txt</parameter>\n"
            "  <parameter=content>line one\nline two\nline three</parameter>\n"
            "</function>"
        )
        result = parse_mcp_tool_calls(text)
        self.assertEqual(len(result), 1)
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args["content"], "line one\nline two\nline three")

    def test_tool_call_id_format(self):
        text = (
            "<function=fn_a>\n  <parameter=x>1</parameter>\n</function>\n"
            "<function=fn_b>\n  <parameter=y>2</parameter>\n</function>"
        )
        result = parse_mcp_tool_calls(text)
        self.assertEqual(result[0]["id"], "call_0")
        self.assertEqual(result[1]["id"], "call_1")
        self.assertEqual(result[0]["type"], "function")
        self.assertEqual(result[1]["type"], "function")

    def test_tool_call_with_json_param_value(self):
        text = (
            "<function=process>\n"
            '  <parameter=data>{\"key\": \"value\", \"num\": 42}</parameter>\n'
            "</function>"
        )
        result = parse_mcp_tool_calls(text)
        self.assertEqual(len(result), 1)
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args['data'], '{"key": "value", "num": 42}')

    def test_whitespace_tolerance(self):
        text = "  <function = get_weather>  \n    <parameter = city >  Tokyo  </parameter>  \n  </function>  "
        result = parse_mcp_tool_calls(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "get_weather")
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args["city"], "Tokyo")


if __name__ == "__main__":
    unittest.main()
