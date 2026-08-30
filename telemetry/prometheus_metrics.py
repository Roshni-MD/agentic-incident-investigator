from prometheus_client import Gauge


CPU_UTILIZATION = Gauge(
    "ml_service_cpu_utilization",
    "CPU utilization percentage",
    ["service"],
)

GPU_UTILIZATION = Gauge(
    "ml_service_gpu_utilization",
    "GPU utilization percentage",
    ["service"],
)

GPU_MEMORY_UTILIZATION = Gauge(
    "ml_service_gpu_memory_utilization",
    "GPU memory utilization percentage",
    ["service"],
)

P95_LATENCY = Gauge(
    "ml_service_p95_latency_ms",
    "P95 inference latency in milliseconds",
    ["service"],
)

THROUGHPUT = Gauge(
    "ml_service_throughput_rps",
    "Inference throughput requests per second",
    ["service"],
)

DATA_LOADING_LATENCY = Gauge(
    "ml_service_data_loading_ms",
    "Data loading latency in milliseconds",
    ["service"],
)

GPU_KERNEL_TIME = Gauge(
    "ml_service_gpu_kernel_ms",
    "GPU kernel execution time in milliseconds",
    ["service"],
)

NETWORK_UTILIZATION = Gauge(
    "ml_service_network_utilization",
    "Network utilization percentage",
    ["service"],
)

NETWORK_LATENCY = Gauge(
    "ml_service_network_latency_ms",
    "Network latency in milliseconds",
    ["service"],
)


def publish_metrics(
    service_name: str,
    cpu_utilization: float,
    gpu_utilization: float,
    gpu_memory_utilization: float,
    p95_latency_ms: float,
    throughput_rps: float,
    data_loading_ms: float,
    gpu_kernel_ms: float,
    network_utilization: float,
    network_latency_ms: float,
) -> None:

    CPU_UTILIZATION.labels(service_name).set(cpu_utilization)

    GPU_UTILIZATION.labels(service_name).set(gpu_utilization)

    GPU_MEMORY_UTILIZATION.labels(service_name).set(
        gpu_memory_utilization
    )

    P95_LATENCY.labels(service_name).set(
        p95_latency_ms
    )

    THROUGHPUT.labels(service_name).set(
        throughput_rps
    )

    DATA_LOADING_LATENCY.labels(service_name).set(
        data_loading_ms
    )

    GPU_KERNEL_TIME.labels(service_name).set(
        gpu_kernel_ms
    )

    NETWORK_UTILIZATION.labels(service_name).set(
        network_utilization
    )

    NETWORK_LATENCY.labels(service_name).set(
        network_latency_ms
    )