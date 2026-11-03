"""
Integration Bridge: Faucet/Gauge -> ML Service -> Faucet Mitigation
Polls flow stats from Gauge/Prometheus, runs ML detection, applies mitigation via Faucet ACL
"""

import asyncio
import logging
import time
import signal
from typing import Dict, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

import pandas as pd

from .config import settings
from .model import ModelManager, DDoSModel
from .faucet_client import FaucetClient, FlowStat, FaucetConfigManager
from .grpc_service import DDoSServiceServicer

logger = logging.getLogger(__name__)


@dataclass
class SwitchState:
    """Track state per switch"""
    switch_id: str
    last_poll: float = 0
    flow_count: int = 0
    ddos_count: int = 0
    benign_count: int = 0
    last_mitigation: Dict[int, float] = field(default_factory=dict)  # port -> timestamp
    blocked_ports: Set[int] = field(default_factory=set)
    victim_history: List[str] = field(default_factory=list)


class FaucetMLBridge:
    """
    Main integration bridge connecting:
    - Gauge/Prometheus (flow stats collection)
    - ML Service (DDoS detection)
    - Faucet (mitigation via ACL)
    """

    def __init__(self):
        self.model_manager = ModelManager(settings.model_dir)
        self.faucet_client = FaucetClient()
        self.config_manager = FaucetConfigManager()
        self.grpc_servicer = DDoSServiceServicer(self.model_manager, self.faucet_client)

        # State tracking
        self.switches: Dict[str, SwitchState] = {}
        self.running = False
        self.poll_interval = settings.stats_poll_interval

        # Statistics
        self.stats = {
            'polls': 0,
            'predictions': 0,
            'attacks_detected': 0,
            'mitigations': 0,
            'errors': 0
        }

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    async def initialize(self):
        """Initialize components"""
        logger.info("Initializing Faucet-ML Bridge...")

        # Load models
        self.model_manager.load_all()

        if not self.model_manager.current_model:
            logger.warning("No model loaded, attempting auto-train...")
            await self._auto_train()

        # Initialize switch states from config
        await self._discover_switches()

        logger.info("Bridge initialized successfully")

    async def _discover_switches(self):
        """Discover switches from Faucet/Gauge"""
        try:
            # Query Prometheus for known switches
            response = await self.faucet_client.client.get(
                f"{self.faucet_client.gauge_url}/api/v1/query",
                params={'query': 'of_dp_status'}
            )

            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    for result in data['data']['result']:
                        dp_name = result['metric'].get('dp_name', '')
                        if dp_name:
                            self.switches[dp_name] = SwitchState(switch_id=dp_name)
                            logger.info(f"Discovered switch: {dp_name}")

        except Exception as e:
            logger.error(f"Switch discovery failed: {e}")

    async def _auto_train(self):
        """Auto-train model if dataset exists"""
        import os
        if os.path.exists(settings.dataset_path):
            try:
                logger.info(f"Training model from {settings.dataset_path}")
                import pandas as pd
                df = pd.read_csv(settings.dataset_path)
                self.model_manager.train_model(settings.default_model, df)
                logger.info("Auto-train completed")
            except Exception as e:
                logger.error(f"Auto-train failed: {e}")

    async def run(self):
        """Main bridge loop"""
        self.running = True
        logger.info(f"Starting bridge with poll interval: {self.poll_interval}s")

        while self.running:
            start_time = time.time()

            try:
                await self._poll_and_process()
                self.stats['polls'] += 1
            except Exception as e:
                logger.error(f"Poll cycle error: {e}")
                self.stats['errors'] += 1

            # Sleep until next interval
            elapsed = time.time() - start_time
            sleep_time = max(0, self.poll_interval - elapsed)
            await asyncio.sleep(sleep_time)

        logger.info("Bridge stopped")
        await self._cleanup()

    async def _poll_and_process(self):
        """Poll flow stats and process with ML"""
        # Get flow stats from Gauge/Prometheus
        flows = await self.faucet_client.get_flow_stats()

        if not flows:
            logger.debug("No flows received")
            return

        # Group flows by switch
        flows_by_switch = defaultdict(list)
        for flow in flows:
            flows_by_switch[flow.switch_id].append(flow)

        # Process each switch
        for switch_id, switch_flows in flows_by_switch.items():
            if switch_id not in self.switches:
                self.switches[switch_id] = SwitchState(switch_id=switch_id)

            await self._process_switch_flows(switch_id, switch_flows)

    async def _process_switch_flows(self, switch_id: str, flows: List[FlowStat]):
        """Process flows for a single switch"""
        state = self.switches[switch_id]

        # Convert to DataFrame for ML
        flow_data = []
        for flow in flows:
            flow_data.append({
                'timestamp': flow.timestamp,
                'datapath_id': hash(switch_id) % 1000000,  # Use hash as datapath_id
                'flow_id': flow.flow_id,
                'ip_src': flow.ipv4_src,
                'tp_src': flow.tcp_src or flow.udp_src or 0,
                'ip_dst': flow.ipv4_dst,
                'tp_dst': flow.tcp_dst or flow.udp_dst or 0,
                'ip_proto': flow.ip_proto,
                'icmp_code': flow.icmpv4_code or 0,
                'icmp_type': flow.icmpv4_type or 0,
                'flow_duration_sec': flow.duration_sec,
                'flow_duration_nsec': flow.duration_nsec,
                'idle_timeout': flow.idle_timeout,
                'hard_timeout': flow.hard_timeout,
                'flags': 0,  # Not available from Gauge
                'packet_count': flow.packet_count,
                'byte_count': flow.byte_count,
                'packet_count_per_second': flow.packet_count / max(flow.duration_sec, 1),
                'packet_count_per_nsecond': 0,
                'byte_count_per_second': flow.byte_count / max(flow.duration_sec, 1),
                'byte_count_per_nsecond': 0
            })

        if not flow_data:
            return

        df = pd.DataFrame(flow_data)

        # Run prediction
        try:
            labels, confidences = self.model_manager.predict(df)
            self.stats['predictions'] += len(labels)
        except Exception as e:
            logger.error(f"Prediction failed for {switch_id}: {e}")
            return

        # Analyze results
        ddos_flows = []
        benign_flows = []

        for i, (label, conf) in enumerate(zip(labels, confidences)):
            flow = flows[i]
            if label == 1:
                ddos_flows.append((flow, conf))
            else:
                benign_flows.append((flow, conf))

        ddos_count = len(ddos_flows)
        benign_count = len(benign_flows)
        total = ddos_count + benign_count

        state.flow_count = total
        state.ddos_count = ddos_count
        state.benign_count = benign_count
        state.last_poll = time.time()

        # Calculate DDoS percentage
        ddos_pct = (ddos_count / total * 100) if total > 0 else 0

        logger.info(f"{switch_id}: {total} flows, {ddos_count} DDoS ({ddos_pct:.1f}%), {benign_count} benign")

        # Check for attack
        if ddos_pct > settings.ddos_threshold_percent and ddos_count > 0:
            await self._handle_attack(switch_id, ddos_flows, ddos_pct)

        # Update flow count metric
        self.faucet_client.update_flow_count(switch_id, total)

    async def _handle_attack(self, switch_id: str, ddos_flows: List, ddos_pct: float):
        """Handle detected DDoS attack"""
        self.stats['attacks_detected'] += 1

        state = self.switches[switch_id]

        # Determine victim and attack source ports
        victim_ports = defaultdict(int)
        source_ports = defaultdict(int)

        for flow, conf in ddos_flows:
            # Track victim (destination) ports
            if flow.in_port:
                victim_ports[flow.in_port] += flow.packet_count

            # Track source (could be from other switches)
            source_ports[flow.ipv4_src] += flow.packet_count

        # Find primary victim port
        if victim_ports:
            victim_port = max(victim_ports, key=victim_ports.get)
            victim_host = f"port_{victim_port}"

            # Check cooldown
            now = time.time()
            if victim_port in state.last_mitigation:
                elapsed = now - state.last_mitigation[victim_port]
                if elapsed < settings.mitigation_cooldown:
                    logger.info(f"Cooldown active for {switch_id}:{victim_port}, skipping mitigation")
                    return

            # Apply mitigation
            logger.warning(f"ATTACK DETECTED on {switch_id}:{victim_port} ({ddos_pct:.1f}% DDoS)")
            logger.warning(f"Applying mitigation: BLOCK_PORT")

            success = await self.faucet_client.apply_acl(switch_id, victim_port, block=True)

            if success:
                state.blocked_ports.add(victim_port)
                state.last_mitigation[victim_port] = now
                state.victim_history.append(f"{victim_port}:{datetime.now().isoformat()}")
                self.stats['mitigations'] += 1
                logger.info(f"Mitigation applied: {switch_id}:{victim_port} blocked")
            else:
                logger.error(f"Mitigation failed: {switch_id}:{victim_port}")

    async def _cleanup(self):
        """Cleanup on shutdown"""
        logger.info("Cleaning up...")
        await self.faucet_client.close()
        logger.info("Cleanup complete")

    def get_status(self) -> Dict:
        """Get bridge status"""
        return {
            'running': self.running,
            'poll_interval': self.poll_interval,
            'switches': {
                sid: {
                    'switch_id': s.switch_id,
                    'last_poll': s.last_poll,
                    'flow_count': s.flow_count,
                    'ddos_count': s.ddos_count,
                    'benign_count': s.benign_count,
                    'blocked_ports': list(s.blocked_ports),
                    'victim_history': s.victim_history[-10:]  # Last 10
                }
                for sid, s in self.switches.items()
            },
            'stats': self.stats,
            'model_loaded': self.model_manager.current_model is not None,
            'current_model': self.model_manager.current_model.model_type if self.model_manager.current_model else None
        }


async def main():
    """Main entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    bridge = FaucetMLBridge()
    await bridge.initialize()
    await bridge.run()


if __name__ == "__main__":
    asyncio.run(main())