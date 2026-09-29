"""
Mission Flight Data Recorder (FDR) and Replay Engine
Records telemetry and digital twin states to disk and enables post-flight mission replay
with time-scrubbing, speed control (0.5x - 10x), and step-by-step diagnostic inspection.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..core.state import EngineFullStatePacket

class MissionRecorderReplay:
    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = str(Path(__file__).parent.parent.parent / "recordings")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_session_file: Optional[Path] = None
        self.recording_buffer: List[Dict[str, Any]] = []
        
        # Replay state
        self.is_replaying = False
        self.replay_data: List[Dict[str, Any]] = []
        self.replay_cursor = 0
        self.replay_speed = 1.0

    def start_recording(self, mission_name: str = "mission") -> str:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{mission_name}_{timestamp_str}.jsonl"
        self.active_session_file = self.storage_dir / filename
        self.recording_buffer = []
        return str(self.active_session_file)

    def record_frame(self, packet: EngineFullStatePacket):
        if self.active_session_file:
            data = packet.model_dump()
            with open(self.active_session_file, "a") as f:
                f.write(json.dumps(data) + "\n")

    def list_recorded_missions(self) -> List[Dict[str, Any]]:
        missions = []
        for p in sorted(self.storage_dir.glob("*.jsonl"), reverse=True):
            stat = p.stat()
            missions.append({
                "filename": p.name,
                "filepath": str(p),
                "size_kb": round(stat.st_size / 1024, 1),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime))
            })
        return missions

    def load_for_replay(self, mission_filename: str) -> bool:
        p = self.storage_dir / mission_filename
        if not p.exists():
            return False
        self.replay_data = []
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.replay_data.append(json.loads(line))
        self.replay_cursor = 0
        self.is_replaying = True
        return True

    def get_replay_progress(self) -> Dict[str, Any]:
        total = len(self.replay_data)
        return {
            "is_replaying": self.is_replaying,
            "cursor": self.replay_cursor,
            "total_frames": total,
            "pct_complete": round((self.replay_cursor / max(1, total)) * 100.0, 1),
            "speed": self.replay_speed
        }

    def scrub_to(self, frame_index: int) -> Optional[Dict[str, Any]]:
        if not self.replay_data:
            return None
        self.replay_cursor = max(0, min(len(self.replay_data) - 1, frame_index))
        return self.replay_data[self.replay_cursor]

    def step_replay(self) -> Optional[Dict[str, Any]]:
        if not self.replay_data or self.replay_cursor >= len(self.replay_data):
            return None
        frame = self.replay_data[self.replay_cursor]
        self.replay_cursor += 1
        return frame
