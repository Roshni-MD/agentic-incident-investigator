import pytest
from datetime import datetime, timezone

from telemetry.query import normalize_query_result

from telemetry.query import (
    PrometheusClient,
    get_metric_history,
    build_metric_query,
    get_metric_name,
)


def test_metric_mapping():
    assert (
        get_metric_name("gpu_utilization")
        == "ml_service_gpu_utilization"
    )


def test_build_metric_query():
    query = build_metric_query(
        metric_name="gpu_utilization",
        service_name="image-ranking-service",
    )

    assert query == (
        'ml_service_gpu_utilization'
        '{service="image-ranking-service"}'
    )


def test_unknown_metric_rejected():
    with pytest.raises(ValueError):
        get_metric_name("temperature")


def test_normalize_instant_query():

    data = {
        "resultType": "vector",
        "result": [
            {
                "metric": {
                    "service": "image-ranking-service",
                },
                "value": [
                    1000.0,
                    "42",
                ],
            }
        ],
    }

    points = normalize_query_result(data)

    assert len(points) == 1
    assert points[0].value == 42.0
    assert points[0].timestamp == datetime.fromtimestamp(
        1000.0,
        tz=timezone.utc,
    )


def test_normalize_range_query():

    data = {
        "resultType": "matrix",
        "result": [
            {
                "metric": {
                    "service": "image-ranking-service",
                },
                "values": [
                    [1000.0, "55"],
                    [1001.0, "60"],
                    [1002.0, "96"],
                ],
            }
        ],
    }

    points = normalize_query_result(data)

    assert len(points) == 3
    assert [p.value for p in points] == [
        55.0,
        60.0,
        96.0,
    ]

def test_get_metric_history(monkeypatch):

    expected_data = {
        "resultType": "matrix",
        "result": [
            {
                "metric": {
                    "service": "image-ranking-service",
                },
                "values": [
                    [1000.0, "91"],
                    [1001.0, "91"],
                    [1002.0, "42"],
                ],
            }
        ],
    }

    class FakeResponse:

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "status": "success",
                "data": expected_data,
            }

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "telemetry.query.requests.get",
        fake_get,
    )

    client = PrometheusClient()

    points = get_metric_history(
        client=client,
        metric_name="gpu_utilization",
        service_name="image-ranking-service",
        start_time=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        end_time=datetime(
            2026,
            1,
            1,
            0,
            1,
            tzinfo=timezone.utc,
        ),
        step_seconds=1,
    )

    assert len(points) == 3

    assert [point.value for point in points] == [
        91.0,
        91.0,
        42.0,
    ]

    assert all(
        point.timestamp.tzinfo == timezone.utc
        for point in points
    )

def test_invalid_service_name_rejected():

    with pytest.raises(ValueError):

        build_metric_query(
            metric_name="gpu_utilization",
            service_name='foo"}[5m]',
        )