from __future__ import annotations

from .image import ImageClient
from .mock import MockImageClient, MockSamClient, MockSparkClient
from .sam import SamClient
from .spark import SparkClient


def get_clients(config) -> tuple[SparkClient, ImageClient, SamClient]:
    if config.get("mock", {}).get("enabled"):
        mock_config = config.get("mock", {})
        return (
            MockSparkClient(),
            MockImageClient(),
            MockSamClient(
                force_fail_first_round=bool(
                    mock_config.get("force_fail_first_round", False)
                )
            ),
        )
    return SparkClient(config), ImageClient(config), SamClient(config)
