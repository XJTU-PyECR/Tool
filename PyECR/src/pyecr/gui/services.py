"""Service contracts and mock implementations for the GUI shell."""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Protocol

from pyecr.gui.state import GuiPage, GuiSnapshot
from pyecr.gui.worker import ProgressCallback

__all__ = ["GuiService", "MockGuiService"]


class GuiService(Protocol):
    """Interface implemented later by scientific service adapters."""

    def execute(
        self,
        page: GuiPage,
        context: GuiSnapshot,
        progress_callback: ProgressCallback,
    ) -> Mapping[str, object]:
        """Execute one page action while reporting normalized progress."""


class MockGuiService:
    """Deterministic placeholder used by every first-generation Run button."""

    def execute(
        self,
        page: GuiPage,
        context: GuiSnapshot,
        progress_callback: ProgressCallback,
    ) -> Mapping[str, object]:
        progress_callback(0.0, f"Preparing mock {page.value} service")
        progress_callback(0.5, "Mock work in progress")
        progress_callback(1.0, "Mock result ready")
        return {
            "service": "mock",
            "page": page.value,
            "message": "No scientific calculation was performed.",
            "context": asdict(context),
        }

