import itertools
import threading
from typing import Dict, List

import requests


class RMBGServiceClient:
    def __init__(self, base_url: str, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> bool:
        resp = requests.get(f"{self.base_url}/health", timeout=5)
        return resp.status_code == 200

    def remove(self, image_base64: str) -> str:
        resp = requests.post(
            f"{self.base_url}/remove",
            json={"image": image_base64},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["image"]


class RMBGServicePool:
    def __init__(self, endpoints: List[str], timeout: int = 60) -> None:
        if len(endpoints) == 0:
            raise ValueError("At least one RMBG endpoint is required")
        self.clients = [RMBGServiceClient(url, timeout=timeout) for url in endpoints]
        self._lock = threading.Lock()
        self._cursor = itertools.cycle(range(len(self.clients)))

    def remove(self, image_base64: str) -> str:
        with self._lock:
            idx = next(self._cursor)
        return self.clients[idx].remove(image_base64)

    def health(self) -> Dict[str, bool]:
        status: Dict[str, bool] = {}
        for client in self.clients:
            try:
                status[client.base_url] = client.health()
            except Exception:
                status[client.base_url] = False
        return status
