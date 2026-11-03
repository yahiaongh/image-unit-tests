from __future__ import annotations

from .image import ImageClient
from .mock import MockImageClient, MockSamClient, MockSparkClient
from .sam import SamClient
from .spark import SparkClient


def get_clients(config) -> tuple[SparkClient, ImageClient, SamClient]:
    mock_config = config.get("mock", {})
    use_mock_spark = mock_config.get("spark", mock_config.get("enabled", False))
    use_mock_image = mock_config.get("image", mock_config.get("enabled", False))
    use_mock_sam = mock_config.get("sam", mock_config.get("enabled", False))
    spark = MockSparkClient() if use_mock_spark else SparkClient(config)
    image = MockImageClient(config) if use_mock_image else ImageClient(config)
    sam = (
        MockSamClient(
            config,
            force_fail_first_round=bool(
                mock_config.get("force_fail_first_round", False)
            ),
        )
        if use_mock_sam
        else SamClient(config)
    )
    return spark, image, sam
