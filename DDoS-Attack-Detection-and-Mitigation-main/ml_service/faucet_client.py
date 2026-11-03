"""
Faucet Controller Client for ML Service Integration
Communicates with Faucet via REST API and Prometheus metrics
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import httpx
import yaml
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class FlowStat:
    """Flow statistics from Faucet/Gauge"""
    switch_id: str
    flow_id: str
    in_port: int
    eth_type: int
    ipv4_src: str
    ipv4_dst: str
    ip_proto: int
    tcp_src: Optional[int] = None
    tcp_dst: Optional[int] = None
    udp_src: Optional[int] = None
    udp_dst: Optional[int] = None
    icmpv4_type: Optional[int] = None
    icmpv4_code: Optional[int] = None
    packet_count: int = 0
    byte_count: int = 0
    duration_sec: int = 0
    duration_nsec: int = 0
    idle_timeout: int = 0
    hard_timeout: int = 0
    priority: int = 0
    cookie: int = 0
    timestamp: float = 0


@dataclass
class PortStat:
    """Port statistics from Faucet/Gauge"""
    switch_id: str
    port_number: int
    port_name: str
    tx_packets: int = 0
    rx_packets: int = 0
    tx_bytes: int = 0
    rx_bytes: int = 0
    tx_errors: int = 0
    rx_errors: int = 0
    dropped: int = 0
    timestamp: float = 0


class FaucetClient:
    """Client for communicating with Faucet controller"""

    def __init__(self):
        self.faucet_url = f"http://{settings.faucet_host}:{settings.faucet_rest_port}"
        self.gauge_url = f"http://{settings.faucet_host}:{settings.gauge_prometheus_port}"
        self.acl_name = settings.faucet_acl_name
        self.client = httpx.AsyncClient(timeout=30.0)

        # Prometheus metrics for ML service
        self.registry = CollectorRegistry()
        self.predictions_total = Counter(
            'ddos_predictions_total',
            'Total predictions made',
            ['result'],
            registry=self.registry
        )
        self.attack_detected = Counter(
            'ddos_attacks_detected_total',
            'Total DDoS attacks detected',
            ['switch', 'victim'],
            registry=self.registry
        )
        self.mitigation_actions = Counter(
            'ddos_mitigation_actions_total',
            'Total mitigation actions taken',
            ['action', 'switch', 'result'],
            registry=self.registry
        )
        self.prediction_latency = Histogram(
            'ddos_prediction_latency_seconds',
            'Prediction latency in seconds',
            registry=self.registry
        )
        self.flow_count = Gauge(
            'ddos_flows_processed',
            'Number of flows processed',
            ['switch'],
            registry=self.registry
        )

    async def close(self):
        await self.client.aclose()

    async def get_flow_stats(self) -> List[FlowStat]:
        """Fetch flow statistics from Gauge/Prometheus"""
        try:
            # Query Prometheus for flow stats
            queries = {
                'packet_count': 'of_flow_packet_count',
                'byte_count': 'of_flow_byte_count',
                'duration_sec': 'of_flow_duration_seconds',
                'duration_nsec': 'of_flow_duration_nanoseconds',
                'idle_timeout': 'of_flow_idle_timeout',
                'hard_timeout': 'of_flow_hard_timeout',
                'priority': 'of_flow_priority',
                'cookie': 'of_flow_cookie',
                'in_port': 'of_flow_match_in_port',
                'eth_type': 'of_flow_match_eth_type',
                'ipv4_src': 'of_flow_match_ipv4_src',
                'ipv4_dst': 'of_flow_match_ipv4_dst',
                'ip_proto': 'of_flow_match_ip_proto',
                'tcp_src': 'of_flow_match_tcp_src',
                'tcp_dst': 'of_flow_match_tcp_dst',
                'udp_src': 'of_flow_match_udp_src',
                'udp_dst': 'of_flow_match_udp_dst',
                'icmpv4_type': 'of_flow_match_icmpv4_type',
                'icmpv4_code': 'of_flow_match_icmpv4_code',
            }

            flows_by_id = {}

            for metric_name, prom_query in queries.items():
                response = await self.client.get(
                    f"{self.gauge_url}/api/v1/query",
                    params={'query': prom_query}
                )

                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 'success':
                        for result in data['data']['result']:
                            metric = result['metric']
                            value = float(result['value'][1])

                            switch = metric.get('dp_name', metric.get('dp_id', 'unknown'))
                            flow_cookie = metric.get('cookie', '')

                            # Create unique flow key
                            flow_key = f"{switch}:{flow_cookie}"

                            if flow_key not in flows_by_id:
                                flows_by_id[flow_key] = FlowStat(
                                    switch_id=switch,
                                    flow_id=flow_key,
                                    timestamp=time.time()
                                )

                            flow = flows_by_id[flow_key]

                            # Map metric to flow attribute
                            if metric_name in ['packet_count', 'byte_count', 'duration_sec',
                                               'duration_nsec', 'idle_timeout', 'hard_timeout',
                                               'priority', 'cookie']:
                                setattr(flow, metric_name, int(value))
                            elif metric_name in ['in_port', 'eth_type', 'ip_proto']:
                                setattr(flow, metric_name, int(value))
                            elif metric_name in ['ipv4_src', 'ipv4_dst']:
                                # Convert from Prometheus format
                                setattr(flow, metric_name, str(int(value)))
                            elif metric_name in ['tcp_src', 'tcp_dst', 'udp_src', 'udp_dst',
                                                 'icmpv4_type', 'icmpv4_code']:
                                if value > 0:
                                    setattr(flow, metric_name, int(value))

            return list(flows_by_id.values())

        except Exception as e:
            logger.error(f"Failed to get flow stats: {e}")
            return []

    async def get_port_stats(self) -> List[PortStat]:
        """Fetch port statistics"""
        try:
            queries = {
                'tx_packets': 'of_port_tx_packets',
                'rx_packets': 'of_port_rx_packets',
                'tx_bytes': 'of_port_tx_bytes',
                'rx_bytes': 'of_port_rx_bytes',
                'tx_errors': 'of_port_tx_errors',
                'rx_errors': 'of_port_rx_errors',
                'dropped': 'of_port_dropped',
            }

            ports_by_id = {}

            for metric_name, prom_query in queries.items():
                response = await self.client.get(
                    f"{self.gauge_url}/api/v1/query",
                    params={'query': prom_query}
                )

                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 'success':
                        for result in data['data']['result']:
                            metric = result['metric']
                            value = float(result['value'][1])

                            switch = metric.get('dp_name', metric.get('dp_id', 'unknown'))
                            port = metric.get('port', metric.get('port_name', '0'))

                            port_key = f"{switch}:{port}"

                            if port_key not in ports_by_id:
                                ports_by_id[port_key] = PortStat(
                                    switch_id=switch,
                                    port_number=int(port) if port.isdigit() else 0,
                                    port_name=port,
                                    timestamp=time.time()
                                )

                            port_stat = ports_by_id[port_key]
                            setattr(port_stat, metric_name, int(value))

            return list(ports_by_id.values())

        except Exception as e:
            logger.error(f"Failed to get port stats: {e}")
            return []

    async def apply_acl(self, switch_id: str, port: int, block: bool = True) -> bool:
        """Apply ACL to block/unblock port on switch"""
        try:
            # Faucet ACL update via config reload
            # In production, use Faucet REST API or config file update

            acl_rule = {
                "rule": {
                    "in_port": port,
                    "actions": {
                        "allow": not block
                    }
                }
            }

            # Write to ACL file (Faucet watches for changes)
            acl_file = Path(f"/etc/faucet/acls/{self.acl_name}.yaml")

            # Read existing ACL
            acls = []
            if acl_file.exists():
                with open(acl_file) as f:
                    acls = yaml.safe_load(f) or []

            # Update or add rule
            rule_exists = False
            for acl in acls:
                if acl.get('rule', {}).get('in_port') == port:
                    acl['rule']['actions']['allow'] = not block
                    rule_exists = True
                    break

            if not rule_exists:
                acls.append(acl_rule)

            # Write back
            acl_file.parent.mkdir(parents=True, exist_ok=True)
            with open(acl_file, 'w') as f:
                yaml.dump(acls, f)

            # Trigger Faucet reload
            await self.reload_faucet_config()

            self.mitigation_actions.labels(
                action='block' if block else 'unblock',
                switch=switch_id,
                result='success'
            ).inc()

            logger.info(f"{'Blocked' if block else 'Unblocked'} port {port} on {switch_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply ACL: {e}")
            self.mitigation_actions.labels(
                action='block' if block else 'unblock',
                switch=switch_id,
                result='failed'
            ).inc()
            return False

    async def reload_faucet_config(self) -> bool:
        """Trigger Faucet configuration reload"""
        try:
            response = await self.client.post(
                f"{self.faucet_url}/config/reload"
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to reload Faucet config: {e}")
            return False

    async def get_switch_status(self) -> Dict[str, Any]:
        """Get Faucet switch status"""
        try:
            response = await self.client.get(f"{self.faucet_url}/metrics")
            if response.status_code == 200:
                return {"status": "ok", "metrics": response.text[:1000]}
            return {"status": "error"}
        except Exception as e:
            logger.error(f"Failed to get switch status: {e}")
            return {"status": "error", "error": str(e)}

    def record_prediction(self, is_ddos: bool, switch: str = "", victim: str = "") -> None:
        """Record prediction metrics"""
        self.predictions_total.labels(result='ddos' if is_ddos else 'benign').inc()
        if is_ddos and victim:
            self.attack_detected.labels(switch=switch, victim=victim).inc()

    def record_mitigation(self, action: str, switch: str, success: bool) -> None:
        """Record mitigation action"""
        self.mitigation_actions.labels(
            action=action,
            switch=switch,
            result='success' if success else 'failed'
        ).inc()

    def record_latency(self, latency: float) -> None:
        """Record prediction latency"""
        self.prediction_latency.observe(latency)

    def update_flow_count(self, switch: str, count: int) -> None:
        """Update flow count gauge"""
        self.flow_count.labels(switch=switch).set(count)

    def push_metrics(self, gateway: str = "localhost:9091") -> None:
        """Push metrics to Prometheus Pushgateway"""
        try:
            push_to_gateway(gateway, job='ddos_ml_service', registry=self.registry)
        except Exception as e:
            logger.warning(f"Failed to push metrics: {e}")


class FaucetConfigManager:
    """Manage Faucet configuration files"""

    def __init__(self, config_dir: Path = Path("/etc/faucet")):
        self.config_dir = config_dir
        self.acl_dir = config_dir / "acls"
        self.acl_dir.mkdir(parents=True, exist_ok=True)

    def write_acl(self, name: str, rules: List[Dict]) -> bool:
        """Write ACL file"""
        try:
            acl_file = self.acl_dir / f"{name}.yaml"
            with open(acl_file, 'w') as f:
                yaml.dump(rules, f, default_flow_style=False)
            return True
        except Exception as e:
            logger.error(f"Failed to write ACL {name}: {e}")
            return False

    def read_acl(self, name: str) -> List[Dict]:
        """Read ACL file"""
        try:
            acl_file = self.acl_dir / f"{name}.yaml"
            if acl_file.exists():
                with open(acl_file) as f:
                    return yaml.safe_load(f) or []
            return []
        except Exception as e:
            logger.error(f"Failed to read ACL {name}: {e}")
            return []

    def add_block_rule(self, acl_name: str, port: int, switch: str = None) -> bool:
        """Add port block rule to ACL"""
        rules = self.read_acl(acl_name)

        # Check if rule exists
        for rule in rules:
            if rule.get('rule', {}).get('in_port') == port:
                rule['rule']['actions']['allow'] = False
                return self.write_acl(acl_name, rules)

        # Add new rule
        new_rule = {
            "rule": {
                "in_port": port,
                "actions": {"allow": False}
            }
        }
        if switch:
            new_rule["rule"]["dp_name"] = switch

        rules.append(new_rule)
        return self.write_acl(acl_name, rules)

    def remove_block_rule(self, acl_name: str, port: int) -> bool:
        """Remove port block rule from ACL"""
        rules = self.read_acl(acl_name)

        # Remove rule for this port
        rules = [r for r in rules if r.get('rule', {}).get('in_port') != port]

        return self.write_acl(acl_name, rules)

    def generate_faucet_config(self, switches: List[Dict]) -> str:
        """Generate Faucet config from topology"""
        # This would generate faucet.yaml from topology description
        # For now, return template
        return ""