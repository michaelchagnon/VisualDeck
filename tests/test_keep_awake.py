# SPDX-License-Identifier: MPL-2.0
"""keep_awake: a show machine does not sleep while VisualDeck is open (Windows)."""
import inspect

import keep_awake


class _Kernel:
    def __init__(self, result=1):
        self.calls, self.result = [], result

    def SetThreadExecutionState(self, flags):  # noqa: N802 - the Windows name
        self.calls.append(flags)
        return self.result


def test_hold_asks_for_system_and_display_until_released(monkeypatch):
    k = _Kernel()
    monkeypatch.setattr(keep_awake, "_kernel32", lambda: k)
    monkeypatch.setattr(keep_awake, "system_state", lambda: {"display", "system"})
    assert keep_awake.hold() and keep_awake.is_held()
    assert k.calls == [keep_awake.ES_CONTINUOUS | keep_awake.ES_SYSTEM_REQUIRED | keep_awake.ES_DISPLAY_REQUIRED]
    keep_awake.release()
    assert k.calls[-1] == keep_awake.ES_CONTINUOUS and not keep_awake.is_held()   # back to the PC's settings
    keep_awake.release()                                                       # twice: nothing more
    assert len(k.calls) == 2


def test_a_refused_request_or_another_system_is_not_an_error(monkeypatch):
    monkeypatch.setattr(keep_awake, "_kernel32", lambda: _Kernel(result=0))
    assert keep_awake.hold() is False and not keep_awake.is_held()
    monkeypatch.setattr(keep_awake, "_kernel32", lambda: None)                 # not Windows
    assert keep_awake.hold() is False


def test_state_names():
    assert keep_awake.state_names(0x1 | 0x2) == {"system", "display"}
    assert keep_awake.state_names(0) == set()


def test_the_app_holds_the_request_for_its_whole_run():
    import app
    src = inspect.getsource(app.main)
    assert src.index("keep_awake.hold()") < src.index("ControlInterface(media_manager)")
    assert "keep_awake.release()" in src.split("finally:", 1)[1]
