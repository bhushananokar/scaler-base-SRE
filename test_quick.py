"""
Quick smoke test — runs an optimal path through each task and prints scores.
Run from my_env/:  python test_quick.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from server.my_env_environment import IncidentResponseEnvironment
from models import IncidentAction


def step(env, **kwargs):
    obs = env.step(IncidentAction(**kwargs))
    print(f"  [{kwargs['tool']:<25}] reward={obs.reward:+.4f}  done={obs.done}")
    return obs


def task1():
    print("\n=== TASK 1: OOM (Easy) ===")
    env = IncidentResponseEnvironment()
    env.reset(task_id=1)
    step(env, tool="list_alerts")
    step(env, tool="query_logs", service="payment-service", severity="error")
    step(env, tool="read_runbook", topic="payment-service")
    step(env, tool="execute_remediation", action_type="restart", target="payment-service")
    step(env, tool="query_metrics", service="payment-service")
    step(env, tool="update_status", message="Restarted payment-service due to OOM heap exhaustion, monitoring recovery")
    obs = step(env, tool="declare_resolved",
               root_cause="OOM in payment-service TransactionCache unbounded memory leak; restarted service to clear heap")
    print(f"\nFINAL SCORE: {obs.reward:.3f}/1.000")
    # Print just the score lines from the feedback
    for line in obs.content.splitlines():
        if line.startswith(("[D", "Final", "===", "Excell", "Good", "Outstand", "Tip", "Warn")):
            print(" ", line)


def task2():
    print("\n=== TASK 2: Bad Deploy (Medium) ===")
    env = IncidentResponseEnvironment()
    env.reset(task_id=2)
    step(env, tool="list_alerts")
    step(env, tool="query_logs", service="api-gateway", severity="error")
    step(env, tool="check_deploy_history", service="api-gateway")
    step(env, tool="query_metrics", service="api-gateway", metric="error_rate")
    step(env, tool="execute_remediation", action_type="rollback", target="api-gateway")
    step(env, tool="query_metrics", service="api-gateway", metric="error_rate")
    step(env, tool="update_status", message="api-gateway v2.3.1 rolled back, NullPointerException resolved, error rate recovering")
    obs = step(env, tool="declare_resolved",
               root_cause="api-gateway v2.3.1 removed lazy init from RouteHandler causing NullPointerException on all routes; rolled back to v2.3.0")
    print(f"\nFINAL SCORE: {obs.reward:.3f}/1.000")
    for line in obs.content.splitlines():
        if line.startswith(("[D", "Final", "===", "Excell", "Good", "Outstand", "Tip", "Warn")):
            print(" ", line)


def task3():
    print("\n=== TASK 3: Cascade (Hard) ===")
    env = IncidentResponseEnvironment()
    env.reset(task_id=3)
    step(env, tool="list_alerts")
    step(env, tool="query_logs", service="payment-service", severity="error")
    step(env, tool="query_logs", service="db-pool", severity="error")
    step(env, tool="query_metrics", service="db-pool", metric="all")
    step(env, tool="query_metrics", service="order-service", metric="db_connections_acquired")
    step(env, tool="check_deploy_history", service="order-service")
    step(env, tool="update_status", message="Root cause: order-service v4.2.0 connection leak exhausting db-pool. Restarting order-service and scaling db-pool.")
    step(env, tool="execute_remediation", action_type="restart", target="order-service")
    step(env, tool="execute_remediation", action_type="scale", target="db-pool")
    step(env, tool="query_metrics", service="payment-service", metric="error_rate")
    obs = step(env, tool="declare_resolved",
               root_cause="order-service v4.2.0 async query batching caused DB connection leak exhausting db-pool; restarted order-service stopped the leak, scaled db-pool cleared the wait queue and restored all downstream services")
    print(f"\nFINAL SCORE: {obs.reward:.3f}/1.000")
    for line in obs.content.splitlines():
        if line.startswith(("[D", "Final", "===", "Excell", "Good", "Outstand", "Tip", "Warn")):
            print(" ", line)


def task4():
    print("\n=== TASK 4: Config Drift / TLS Cert CN Mismatch (Medium-Hard) ===")
    env = IncidentResponseEnvironment()
    env.reset(task_id=4)
    step(env, tool="list_alerts")
    step(env, tool="query_logs", service="checkout-service", severity="error")
    step(env, tool="query_logs", service="payments-gateway", severity="all")
    step(env, tool="check_deploy_history", service="payments-gateway")
    step(env, tool="read_runbook", topic="tls")
    step(env, tool="execute_remediation", action_type="update_config", target="checkout-service")
    step(env, tool="query_metrics", service="checkout-service")
    step(env, tool="update_status", message="TLS CN mismatch: payments-gateway cert rotated, checkout-service config updated to new CN payments-gw-v2.internal")
    obs = step(env, tool="declare_resolved",
               root_cause="payments-gateway TLS certificate rotated at 09:12 UTC changing CN from payments-gw-v1.internal to payments-gw-v2.internal; checkout-service config updated via update_config to reference new CN")
    print(f"\nFINAL SCORE: {obs.reward:.3f}/1.000")
    for line in obs.content.splitlines():
        if line.startswith(("[D", "Final", "===", "Excell", "Good", "Outstand", "Tip", "Warn")):
            print(" ", line)


def task5():
    print("\n=== TASK 5: Alert Storm / Consumer Deadlock (Hard) ===")
    env = IncidentResponseEnvironment()
    env.reset(task_id=5)
    step(env, tool="list_alerts")
    step(env, tool="query_logs", service="message-queue", severity="error")
    step(env, tool="query_logs", service="notification-service", severity="error")
    step(env, tool="check_deploy_history", service="notification-service")
    step(env, tool="query_metrics", service="message-queue")
    step(env, tool="read_runbook", topic="notification-service")
    step(env, tool="update_status", message="Root cause: notification-service v2.8.0 deadlock consuming 0 msgs/min. Fixing consumer first then clearing queue.")
    step(env, tool="execute_remediation", action_type="restart", target="notification-service")
    step(env, tool="execute_remediation", action_type="restart", target="message-queue")
    step(env, tool="query_metrics", service="message-queue")
    obs = step(env, tool="declare_resolved",
               root_cause="notification-service v2.8.0 JPA transaction deadlock blocked all consumer threads causing message-queue to fill (847K depth, 97% memory) and 4 producer services to back up; restarted notification-service to clear deadlock then restarted message-queue to drain queue and restore producers")
    print(f"\nFINAL SCORE: {obs.reward:.3f}/1.000")
    for line in obs.content.splitlines():
        if line.startswith(("[D", "Final", "===", "Excell", "Good", "Outstand", "Tip", "Warn")):
            print(" ", line)


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()
    print("\n\nAll tasks complete.")
