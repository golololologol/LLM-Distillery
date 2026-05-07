from classes.formatter import SentinelFormatter


def test_round_trip(tokenizer):
    fmt = SentinelFormatter(tokenizer, supports_reasoning=False, supports_tool_calls=True)
    fmt.round_trip_check()


def test_round_trip_tiny(tokenizer):
    fmt = SentinelFormatter(tokenizer, supports_reasoning=False, supports_tool_calls=True)
    fmt.round_trip_check()
