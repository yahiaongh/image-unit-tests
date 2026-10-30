from unittest.mock import MagicMock, patch

from src.clients.spark import SparkClient
from src.config import load_config


def test_system_prompt_is_sent():
    cfg = load_config()
    with patch("src.clients.spark.requests.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200,
            content=b'{"choices":[{"message":{"content":"{\\"constraints\\":[]}"}}]}',
            text='{"choices":[{"message":{"content":"{\\"constraints\\":[]}"}}]}',
            json=lambda: {"choices": [{"message": {"content": '{"constraints":[]}'} }]},
        )
        SparkClient(cfg).generate_constraints("test brief")
    payload = mock_post.call_args.kwargs["json"]
    assert any(message["role"] == "system" for message in payload["messages"]), (
        "system prompt missing from request"
    )
    assert payload.get("temperature") == 0.0
    assert payload.get("response_format") == {"type": "json_object"}
    assert payload.get("max_tokens") == 8192


def test_truncated_response_is_retried_once():
    cfg = load_config()
    truncated = MagicMock(
        status_code=200,
        text="response",
        json=lambda: {
            "choices": [{
                "finish_reason": "length",
                "message": {"content": ""},
            }],
            "usage": {"completion_tokens": 8192},
        },
    )
    complete = MagicMock(
        status_code=200,
        text='{"choices":[{"message":{"content":"{\\"constraints\\":[]}"}}]}',
        json=lambda: {"choices": [{
            "finish_reason": "stop",
            "message": {"content": '{"constraints":[]}'} ,
        }]},
    )

    with patch(
        "src.clients.spark.requests.post", side_effect=[truncated, complete]
    ) as mock_post:
        SparkClient(cfg).generate_constraints("test brief")

    assert mock_post.call_count == 2
    assert all(
        call.kwargs["json"]["max_tokens"] == 8192
        for call in mock_post.call_args_list
    )
