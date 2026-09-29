"""
Secure Telemetry Architecture and Cryptographic Integrity Layer
Implements HMAC-SHA256 message authentication, rolling cryptographic sequence numbers,
and anti-replay validation to secure UAV telemetry against electronic tampering and spoofing.
"""

import hmac
import hashlib
import json
import time
from typing import Dict, Any, Tuple, Optional

class SecureTelemetryGateway:
    """
    Secures telemetry streams across tactical line-of-sight (LOS) and satellite data links (SATCOM).
    Uses HMAC-SHA256 with rolling monotonic sequence counters to eliminate replay attacks.
    """
    def __init__(self, pre_shared_key: bytes = b"DRDO_MALE_UAV_PROPULSION_SECURE_KEY_2026"):
        self.psk = pre_shared_key
        self.outbound_seq = 0
        self.last_received_seq = -1

    def sign_packet(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Signs telemetry payload with rolling sequence number and HMAC-SHA256 signature.
        """
        self.outbound_seq += 1
        signed_frame = {
            "seq": self.outbound_seq,
            "tx_timestamp": time.time(),
            "payload": payload
        }
        # Canonical serialized byte representation for deterministic signature
        serialized = json.dumps(signed_frame, sort_keys=True).encode("utf-8")
        signature = hmac.new(self.psk, serialized, hashlib.sha256).hexdigest()
        
        return {
            "secure_envelope": signed_frame,
            "hmac_sha256": signature
        }

    def verify_and_unpack(self, envelope: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Validates cryptographic authenticity and freshness of received telemetry packet.
        Returns: (is_valid, payload, status_message)
        """
        if "secure_envelope" not in envelope or "hmac_sha256" not in envelope:
            return False, None, "INVALID_FRAME_STRUCTURE"

        signed_frame = envelope["secure_envelope"]
        expected_sig = envelope["hmac_sha256"]

        # 1. Verify HMAC-SHA256 signature
        serialized = json.dumps(signed_frame, sort_keys=True).encode("utf-8")
        computed_sig = hmac.new(self.psk, serialized, hashlib.sha256).hexdigest()

        if not hmac.compare_digest(expected_sig, computed_sig):
            return False, None, "CRYPTOGRAPHIC_SIGNATURE_MISMATCH_TAMPERING_DETECTED"

        # 2. Anti-Replay Check (Sequence must be strictly increasing)
        rx_seq = signed_frame.get("seq", 0)
        if rx_seq <= self.last_received_seq:
            return False, None, f"REPLAY_ATTACK_DETECTED: Received seq {rx_seq} <= last seq {self.last_received_seq}"

        # 3. Freshness Check (Packet age within tolerance window of 5 seconds)
        age = abs(time.time() - signed_frame.get("tx_timestamp", 0))
        if age > 5.0:
            return False, None, f"STALE_TELEMETRY_LATENCY_EXCEEDED: Age {age:.2f}s"

        self.last_received_seq = rx_seq
        return True, signed_frame["payload"], "AUTHENTICATED_AND_VERIFIED"
