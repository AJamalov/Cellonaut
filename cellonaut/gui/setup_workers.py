"""Background workers used by setup and installation checks."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal, Slot

from cellonaut.io.fiji_installation import scan_fiji_installation


class FijiInstallationScanWorker(QObject):
    """Scan Fiji folders without blocking the GUI thread."""

    done_signal = Signal(dict)

    def __init__(self, path_text: str):
        super().__init__()
        self.path_text = str(path_text or "").strip()
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    @Slot()
    def run(self):
        try:
            status = scan_fiji_installation(self.path_text, cancel_requested=self._cancel_event.is_set)
            result = {"path": self.path_text, "status": status, "scan_error": ""}
        except Exception as exc:
            result = {
                "path": self.path_text,
                "status": None,
                "scan_error": "" if self._cancel_event.is_set() else str(exc),
            }
        result["scan_cancelled"] = self._cancel_event.is_set()
        try:
            self.done_signal.emit(result)
        except RuntimeError:
            # The window may close while a large Fiji folder is still being
            # scanned, deleting the signal owner before the result is used.
            pass
