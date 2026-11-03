from unittest.mock import MagicMock, patch

import requests
from meta_sam_parser import ResponsesStreamSourceError

from src.clients.sam import SamClient


def test_transient_sam_stream_source_error_is_retried_once():
    cfg = {
        "meta": {"base_url": "https://example.test", "api_key_env": None},
        "models": {"sam": {"model_id": "sam-3.1", "endpoint": "/responses"}},
    }
    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.__exit__.return_value = None

    parse_attempts = 0

    async def parse_response(_response):
        nonlocal parse_attempts
        parse_attempts += 1
        if parse_attempts == 1:
            raise ResponsesStreamSourceError(
                "next", requests.ConnectionError("stream disconnected")
            )
        return object()

    with (
        patch("src.clients.sam._image_data_url", return_value=("data:image/png;base64,", (1, 1))),
        patch("src.clients.sam.requests.post", return_value=response) as mock_post,
        patch("src.clients.sam._parse_response", side_effect=parse_response),
        patch("src.clients.sam._decode_image_result", return_value=[]) as decode,
    ):
        result = SamClient(cfg).segment(b"image-bytes", "apples")

    assert result == []
    assert mock_post.call_count == 2
    decode.assert_called_once()


def test_non_transient_sam_stream_source_error_is_not_retried():
    cfg = {
        "meta": {"base_url": "https://example.test", "api_key_env": None},
        "models": {"sam": {"model_id": "sam-3.1", "endpoint": "/responses"}},
    }
    response = MagicMock(status_code=200)
    response.__enter__.return_value = response
    response.__exit__.return_value = None
    error = ResponsesStreamSourceError("next", ValueError("invalid event"))

    async def parse_response(_response):
        raise error

    with (
        patch("src.clients.sam._image_data_url", return_value=("data:image/png;base64,", (1, 1))),
        patch("src.clients.sam.requests.post", return_value=response) as mock_post,
        patch("src.clients.sam._parse_response", side_effect=parse_response),
    ):
        try:
            SamClient(cfg).segment(b"image-bytes", "apples")
        except ResponsesStreamSourceError as exc:
            assert exc is error
        else:
            raise AssertionError("Expected non-transient parser source error")

    mock_post.assert_called_once()
