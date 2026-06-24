"""Packed ZMQ stream subscriber used by sim2sim visualization."""

import json

import numpy as np

from gear_sonic.sim2sim.constants import PACKED_ZMQ_HEADER_SIZE


class PackedZMQSubscriber:
    """Non-blocking subscriber for packed ZMQ motion streams."""

    _DTYPE_MAP = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "u8": np.uint8,
        "bool": np.bool_,
    }

    def __init__(self, host: str, port: int, topic: str):
        import zmq

        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.connect(f"tcp://{host}:{port}")
        self._topic = topic
        self._msg = None
        print(f"[PackedZMQSubscriber] Connected to tcp://{host}:{port} (topic: {topic})")

    def _unpack(self, packed_data: bytes) -> dict:
        topic_bytes = self._topic.encode("utf-8")
        if not packed_data.startswith(topic_bytes):
            raise ValueError(f"Message does not start with expected topic '{self._topic}'")

        offset = len(topic_bytes)
        if len(packed_data) < offset + PACKED_ZMQ_HEADER_SIZE:
            raise ValueError(
                f"Packed data too small: {len(packed_data)} < {offset + PACKED_ZMQ_HEADER_SIZE}"
            )

        header_bytes = packed_data[offset : offset + PACKED_ZMQ_HEADER_SIZE]
        null_idx = header_bytes.find(b"\x00")
        if null_idx > 0:
            header_bytes = header_bytes[:null_idx]

        header = json.loads(header_bytes.decode("utf-8"))
        result = {"version": header.get("v", 0), "endian": header.get("endian", "le")}
        if "motion_start_frame" in header:
            result["motion_start_frame"] = int(header["motion_start_frame"])
        current_offset = offset + PACKED_ZMQ_HEADER_SIZE

        for field in header.get("fields", []):
            dtype = self._DTYPE_MAP.get(field["dtype"])
            if dtype is None:
                raise ValueError(f"Unsupported dtype in packed message: {field['dtype']}")
            shape = tuple(field["shape"])
            n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            result[field["name"]] = (
                np.frombuffer(packed_data[current_offset : current_offset + n_bytes], dtype=dtype)
                .reshape(shape)
                .copy()
            )
            current_offset += n_bytes
        return result

    def _poll(self):
        import zmq

        try:
            raw = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return
        self._msg = self._unpack(raw)

    def get_msg(self, clear: bool = True):
        self._poll()
        msg = self._msg
        if clear:
            self._msg = None
        return msg

    def close(self):
        self._socket.close()
        self._ctx.term()

