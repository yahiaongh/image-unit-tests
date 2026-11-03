from unittest.mock import MagicMock, patch

import requests

from src.clients.image import ImageClient


def _client() -> ImageClient:
    return ImageClient({
        "meta": {"base_url": "https://example.test", "api_key_env": None},
        "models": {
            "image": {
                "model_id": "muse-image-1.0",
                "endpoint": "/responses",
                "edit_endpoint": "/responses",
            }
        },
    })


def test_generate_retries_once_after_timeout():
    response = MagicMock(status_code=200, content=b"json")
    response.json.return_value = {"output": [{
        "type": "image_generation_call",
        "result": "aW1hZ2U=",
    }]}

    with (
        patch(
            "src.clients.image.requests.post",
            side_effect=[requests.Timeout(), response],
        ) as post,
        patch("src.clients.image._extract_image", return_value=b"image"),
    ):
        result = _client().generate("a brief")

    assert result == b"image"
    assert post.call_count == 2
    assert all(call.kwargs["timeout"] == 120 for call in post.call_args_list)


def test_edit_retries_once_after_timeout():
    response = MagicMock(status_code=200, content=b"json")
    response.json.return_value = {"output": [{
        "type": "image_generation_call",
        "result": "aW1hZ2U=",
    }]}

    with (
        patch(
            "src.clients.image.requests.post",
            side_effect=[requests.Timeout(), response],
        ) as post,
        patch("src.clients.image._image_data_url", return_value="data:image/png;base64,"),
        patch("src.clients.image._extract_image", return_value=b"image"),
    ):
        result = _client().edit(b"image-bytes", "fix it")

    assert result == b"image"
    assert post.call_count == 2
    assert all(call.kwargs["timeout"] == 120 for call in post.call_args_list)
