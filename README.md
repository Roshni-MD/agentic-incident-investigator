# AI Incident Investigation Agent

## Real-world problem

Imagine you're an ML platform engineer at a company running thousands of training/inference jobs.

A production model suddenly has:

- 35% latency increase
- declining throughput
- GPU utilization dropping from 90% to 40%

Instead of manually checking dashboards, logs, deployment history, configuration, and documentation, an agent investigates the incident autonomously.

## Architecture

```text
                    ┌───────────────────────┐
                    │      User / Slack     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Incident Agent      │
                    │   LangGraph           │
                    └───────────┬───────────┘
                                │
             ┌──────────────────┼──────────────────┐
             │                  │                  │
             ▼                  ▼                  ▼
       Metrics Agent       Log Agent        Deployment Agent
             │                  │                  │
             ▼                  ▼                  ▼
         Prometheus          Logs DB         Model Registry
             │                  │                  │
             └──────────────────┼──────────────────┘
                                │
                                ▼
                         RAG Knowledge Base
                                │
                                ▼
                       Root Cause Analysis
                                │
                                ▼
                     Recommended Remediation
```

## MCP component

This is where MCP becomes meaningful rather than just something you mention on your resume.

Build an MCP server exposing tools such as:

- `get_gpu_metrics()`
- `get_cpu_metrics()`
- `get_memory_metrics()`
- `query_logs()`
- `get_recent_deployments()`
- `get_model_version()`
- `get_training_config()`
- `get_inference_config()`
- `rollback_model()`
- `restart_service()`

The agent does not directly know how Prometheus, your log database, or model registry works.

It discovers and invokes tools through MCP.

## Add RAG

Create a knowledge base containing:

```text
docs/
├── gpu_troubleshooting.md
├── pytorch_performance.md
├── distributed_training.md
├── inference_troubleshooting.md
├── kubernetes_runbooks.md
├── nccl_troubleshooting.md
├── incident_playbooks/
│   ├── gpu_oom.md
│   ├── low_gpu_utilization.md
│   ├── network_bottleneck.md
│   └── model_latency.md
```

Then the agent can reason:

**Signal:** GPU utilization dropped after deployment.

**Retrieve:**
- Relevant runbook: "Low GPU utilization can occur when CPU preprocessing becomes the bottleneck."
- Metrics:
  - GPU utilization = 42%
  - CPU utilization = 96%
  - Data loading time = 380 ms
  - GPU kernel execution = 90 ms

### Agent conclusion

> **Likely root cause:** CPU-side preprocessing bottleneck.
>
> **Confidence:** 0.87
>
> **Evidence:**
> 1. CPU utilization increased 61% → 96%
> 2. GPU utilization decreased 91% → 42%
> 3. Data loading latency increased 4.2x
> 4. Deployment introduced image preprocessing version 2.4
>
> **Recommended action:** Roll back the preprocessing component.

