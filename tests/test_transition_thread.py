# SPDX-License-Identifier: MPL-2.0
"""The fade thread must work from captured routing only (no Tk access)."""
import queue
import time

import control_interface as ci_mod


class FakeCI:
    """Has the render queue only: any Tk access from the thread would raise."""
    def __init__(self):
        self.render_command_queue = queue.Queue()


def test_fade_posts_progress_and_reports_completion_without_tk():
    ci = FakeCI()
    tt = ci_mod.TransitionThread(ci)
    tt.start()
    routing = {"output_mode": "Display 1", "target_monitor": 1, "all_assigned_screens": set(), "gui_monitor": 0}
    tt.start_transition({0: {"path": "a.png"}}, {0: {"path": "b.png"}}, 0.2, target_col=3, routing=routing)
    done = tt.completed.get(timeout=2)
    tt.shutdown()
    tt.join(2)
    cmds = []
    while not ci.render_command_queue.empty():
        cmds.append(ci.render_command_queue.get_nowait())
    assert done == 3
    assert len(cmds) >= 5
    assert all(c["target_monitor"] == 1 and c["output_mode"] == "Display 1" for c in cmds)
    progress = [c["transition_state"]["progress"] for c in cmds]
    assert progress == sorted(progress) and progress[-1] == 1.0
