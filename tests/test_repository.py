from datetime import timedelta

from telemetry.scenarios import load_cpu_bottleneck_scenario


def test_get_metrics_within_time_range():
    incident, repository = load_cpu_bottleneck_scenario()

    start = incident.metrics[0].timestamp
    end = incident.metrics[-1].timestamp

    metrics = repository.get_metrics(
        service_name=incident.service_name,
        start_time=start,
        end_time=end,
    )

    assert len(metrics) == 10


def test_get_logs_within_time_range():
    incident, repository = load_cpu_bottleneck_scenario()

    start = incident.started_at
    end = incident.started_at + timedelta(minutes=10)

    logs = repository.get_logs(
        service_name=incident.service_name,
        start_time=start,
        end_time=end,
    )

    assert len(logs) == 2


def test_get_deployments():
    incident, repository = load_cpu_bottleneck_scenario()

    start = incident.started_at - timedelta(minutes=10)
    end = incident.started_at + timedelta(minutes=10)

    deployments = repository.get_deployments(
        service_name=incident.service_name,
        start_time=start,
        end_time=end,
    )

    assert len(deployments) == 1
    assert deployments[0].model_version == "v2.4"