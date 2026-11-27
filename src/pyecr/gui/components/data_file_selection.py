"""Thread-safe Unicode selection feeding the existing preview/import callbacks."""
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import sys
import traceback

from pyecr.gui.data_file_dialog import choose_data_file
from pyecr.gui.state import GuiPage
from pyecr.gui.theme import TOKENS


class DataFileSelectionMixin:
    def _initialize_data_file_picker(self):
        self._data_picker_running = False
        self._data_picker_ticket = None
        self._data_picker_results = Queue()

    def _data_picker_notice(self, text, *, error=False):
        dpg = self._require_dpg()
        if dpg.does_item_exist("pyecr_data_picker_notice"):
            dpg.set_value("pyecr_data_picker_notice", text)
            dpg.configure_item("pyecr_data_picker_notice", show=bool(text),
                               color=TOKENS.colors.error if error else TOKENS.colors.text)

    def _start_data_file_picker(self):
        if sys.platform != "win32":
            self._require_dpg().show_item("pyecr_data_file_dialog")
            return
        if not hasattr(self, "_data_picker_results"):
            self._initialize_data_file_picker()
        if self._data_picker_running:
            return
        preview = self.controller.session.import_preview
        initial = Path(preview.source.source_path).parent if preview is not None else Path.cwd()
        if not initial.is_dir():
            initial = Path.cwd()
        ticket = object()
        session = self.controller.analysis_session
        revision = session.current_revision
        self._data_picker_ticket = ticket
        self._data_picker_running = True
        self._data_picker_notice("File chooser open — Chinese paths and filenames are supported. Cancel keeps current data.")

        def select():
            try:
                value = choose_data_file(str(initial))
                event = (ticket, session, revision, "selected", value)
            except Exception as exc:
                event = (ticket, session, revision, "error", (exc, traceback.format_exc()))
            self._data_picker_results.put(event)

        Thread(target=select, name="pyecr-data-file-picker", daemon=True).start()

    def _poll_data_file_picker(self):
        if not hasattr(self, "_data_picker_results"):
            return
        try:
            ticket, session, revision, kind, value = self._data_picker_results.get_nowait()
        except Empty:
            pass
        else:
            if ticket is self._data_picker_ticket:
                self._data_picker_running = False
                self._data_picker_ticket = None
                current = self.controller.analysis_session
                if current is not session or current.current_revision != revision:
                    self._data_picker_notice("File selection discarded — the analysis changed. Select the file again.")
                elif kind == "error":
                    exc, details = value
                    self.controller.record_callback_exception(GuiPage.DATA, exc, details)
                    self._data_picker_notice("File chooser unavailable — use Open by full path below.", error=True)
                elif not value:
                    self._data_picker_notice("File selection cancelled; current data and preview are unchanged.")
                else:
                    # Only the DPG/render owner changes controller or widgets.
                    # This still previews only; Import remains an explicit action.
                    self._data_picker_notice("")
                    self._select_data_file(None, {"file_path_name": value}, None)
        dpg = self._require_dpg()
        for tag in ("pyecr_data_select_file", "pyecr_data_preview_path"):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=not self._data_picker_running)

    def _preview_data_file_path(self, sender=None, app_data=None, user_data=None):
        if getattr(self, "_data_picker_running", False):
            return
        text = str(self._require_dpg().get_value("pyecr_data_path_draft")).strip()
        if len(text) >= 2 and text[0] == text[-1] == '"':
            text = text[1:-1]  # Explorer's Copy as path; never re-encode Unicode.
        if not text:
            self._data_picker_notice("Enter the full path to an existing CSV, TXT, XLSX, or XLS file.", error=True)
            return
        self._data_picker_notice("")
        self._select_data_file(None, {"file_path_name": text}, None)
