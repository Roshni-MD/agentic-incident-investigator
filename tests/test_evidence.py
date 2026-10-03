from agent.evidence import EvidenceCollector

def test_evidence_collector():
    collector = EvidenceCollector()

    result = {
        "metric": "cpu_utilization",
        "value": 96,
    }

    evidence = collector.collect(
        tool_name="get_current_metric",
        result=result,
    )

    assert evidence[0].source == "get_current_metric"
    assert evidence[0].description == "cpu_utilization is 96"
    assert evidence[0].value == 96

def test_evidence_collector_current_metric():
    collector = EvidenceCollector()

    result = {
        "service_name": "image-ranking-service",
        "metric_name": "cpu_utilization",
        "value": 96.0,
        "timestamp": "2026-08-29T16:58:51+00:00",
    }

    evidence = collector.collect(
        tool_name="get_current_metric",
        result=result,
    )

    assert len(evidence) == 1
    assert evidence[0].source == "get_current_metric"
    assert (
        evidence[0].description
        == "cpu_utilization for image-ranking-service is 96.0"
    )
    assert evidence[0].value == 96.0

def test_evidence_collector_service_health():
    collector = EvidenceCollector()

    result = {
        "cpu": 96,
        "gpu": 42,
        "gpu_mem": 70,
        "p95": 135,
        "throughput": 760,
        "data_loading": 380,
        "gpu_kernel": 90,
    }

    evidence = collector.collect(
        tool_name="get_service_health",
        result=result,
    )

    assert len(evidence) == 7

    assert evidence[0].source == "get_service_health"
    assert evidence[0].description == "cpu is 96"
    assert evidence[0].value == 96

    assert evidence[1].source == "get_service_health"
    assert evidence[1].description == "gpu is 42"
    assert evidence[1].value == 42

    assert evidence[3].description == "p95 is 135"
    assert evidence[3].value == 135

def test_evidence_collector_metric_history():
    collector = EvidenceCollector()

    result = [
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "value": 91.0,
        },
        {
            "timestamp": "2026-01-01T00:00:01+00:00",
            "value": 91.0,
        },
        {
            "timestamp": "2026-01-01T00:00:02+00:00",
            "value": 42.0,
        },
    ]

    evidence = collector.collect(
        tool_name="get_metric_history",
        result=result,
    )

    assert len(evidence) == 3

    assert evidence[0].source == "get_metric_history"
    assert evidence[0].value == 91.0
    assert (
        evidence[0].description
        == "Metric value is 91.0 at 2026-01-01T00:00:00+00:00"
    )

    assert evidence[2].value == 42.0

def test_collect_query_logs():
    collector = EvidenceCollector()

    result = [
        {
            "timestamp": "2026-08-29T16:58:00+00:00",
            "service_name": "image-ranking-service",
            "level": "ERROR",
            "message": "GPU utilization dropped unexpectedly",
            "metadata": {"request_id": "req-123"},
        },
        {
            "timestamp": "2026-08-29T16:58:01+00:00",
            "service_name": "image-ranking-service",
            "level": "INFO",
            "message": "Retrying inference request",
            "metadata": {},
        },
    ]

    evidence = collector.collect(
        tool_name="query_logs",
        result=result,
    )

    assert len(evidence) == 2

    assert evidence[0].source == "query_logs"
    assert (
        evidence[0].description
        == "ERROR log at 2026-08-29T16:58:00+00:00: "
           "GPU utilization dropped unexpectedly"
    )
    assert evidence[0].value == result[0]

    assert evidence[1].source == "query_logs"
    assert evidence[1].value == result[1]

def test_collect_recent_deployments():
    collector = EvidenceCollector()

    result = [
        {
            "deployment_id": "deploy-123",
            "service_name": "image-ranking-service",
            "model_name": "ranking-model",
            "model_version": "v2",
            "timestamp": "2026-08-29T16:55:00+00:00",
            "previous_version": "v1",
        }
    ]

    evidence = collector.collect(
        tool_name="get_recent_deployments",
        result=result,
    )

    assert len(evidence) == 1

    assert evidence[0].source == "get_recent_deployments"

    assert (
        evidence[0].description
        == "Deployment deploy-123 for image-ranking-service: "
           "ranking-model changed from v1 to v2 "
           "at 2026-08-29T16:55:00+00:00"
    )

    assert evidence[0].value == result[0]