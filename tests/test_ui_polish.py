# SPDX-License-Identifier: MPL-2.0
"""Shortcuts, Help and theme: the table, the bindings (with real Tk key events) and the help text."""
import os
import re

import pytest

import help_content
import shortcuts
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401

tk = pytest.importorskip("tkinter")


def test_every_shortcut_has_a_handler_and_unique_keys(blind_ci):
    acts = blind_ci._shortcut_actions()
    seqs = [seq for s in shortcuts.SHORTCUTS for seq in s.sequences]
    assert len(seqs) == len(set(seqs)), "two shortcuts share a key"
    for s in shortcuts.SHORTCUTS:
        assert s.action == "go" or s.action in acts, s.action


def test_help_lists_every_shortcut_and_the_guide_is_current():
    md = help_content.to_markdown()
    for s in shortcuts.SHORTCUTS:
        if not s.action.startswith("goto_"):
            assert f"| {s.label} |" in md, s.label
    assert "| Ctrl+1 … Ctrl+9 |" in md
    guide = os.path.join(os.path.dirname(__file__), "..", "docs", "USER_GUIDE.md")
    assert open(guide, encoding="utf-8").read() == md, "regenerate: python src/help_content.py > docs/USER_GUIDE.md"


def test_help_names_real_menu_items():
    import inspect
    import control_interface
    src = inspect.getsource(control_interface)
    text = " ".join(help_content.topic_text(blocks) for _, blocks in help_content.TOPICS)
    for item in re.findall(r"› ([A-Z][\w ]+?…?)(?:[ ,.(]|$)", text):
        item = item.strip()
        if item in ("Off", "Editing the live cue", "Remote control", "Apply immediately", "VisualDeck Help",
                    "Test Pattern", "Opacity", "Blend") or item.startswith(("Add Media", "Output")):
            continue
        assert item in src, f"help mentions '{item}', which the UI does not have"



def test_help_does_not_offer_hidden_features():
    import control_interface as ci_mod
    md = help_content.to_markdown()
    if not ci_mod.VD_SHOW_VIDEO_OPTIMIZE_MENU:
        assert "Optimize" not in md
    if not ci_mod.VD_SHOW_DELETE_FROM_DISK_MENU:
        assert "DELETE FROM DISK" not in md


def test_editor_starts_no_worker_thread(tmp_path):
    """F19: the editor used to start a thread whose first step touched Tk."""
    import threading
    from types import SimpleNamespace
    from PIL import Image
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import editor
    from utils.preview_bridge import PreviewBridge
    p = str(tmp_path / "i.png")
    Image.new("RGB", (40, 30), (9, 9, 9)).save(p)
    root = tk.Tk()
    root.withdraw()
    try:
        before = threading.active_count()
        ed = editor.MediaEditor(SimpleNamespace(root=root), p, None, SimpleNamespace(monitors=[]),
                                preview_bridge=PreviewBridge())
        assert threading.active_count() == before
        ed._cancel_and_close()
    finally:
        root.destroy()


@pytest.fixture
def ui(blind_ci):
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    root = tk.Tk()
    import ui_theme
    ui_theme.apply(root)
    blind_ci.root = root
    calls = []
    blind_ci._shortcut_actions = lambda: {a: (lambda a=a: calls.append(a)) for a in
                                          [s.action for s in shortcuts.SHORTCUTS]}
    blind_ci._bind_shortcuts()
    blind_ci.calls = calls
    yield blind_ci
    root.destroy()


def press(widget, seq):
    widget.focus_force()
    widget.update()
    widget.event_generate(seq, when="now")
    widget.update()


def test_key_events_trigger_actions_but_not_while_typing(ui):
    btn = tk.Button(ui.root, text="x")
    btn.pack()
    entry = tk.Entry(ui.root)
    entry.pack()
    for seq in ("<Control-b>", "<Control-B>", "<Shift-space>", "<Control-Key-3>", "<Control-comma>",
                "<Control-Shift-R>", "<F1>"):
        press(btn, seq)
    assert ui.calls == ["blackout", "blackout", "back", "goto_3", "preferences", "relink", "help"]
    ui.calls.clear()
    entry.insert(0, "abc")
    press(entry, "<Control-b>")            # Tk's own "cursor left" in a text field
    press(entry, "<Control-h>")            # Tk's own "delete"
    press(entry, "<F1>")                   # Help works everywhere
    # The field kept its editing keys. Ctrl+B is "cursor left" only where Tk binds it
    # (X11 and macOS, not Windows): then it moved left of "c" and Ctrl+H deleted "b".
    ctrl_b_moves = "<Control-Key-b>" in ui.root.event_info("<<PrevChar>>")
    assert ui.calls == ["help"] and entry.get() == ("ac" if ctrl_b_moves else "ab")


def test_theme_applies_fonts_and_styles(ui):
    import tkinter.font as tkfont
    import ui_theme
    from tkinter import ttk
    assert tkfont.nametofont("TkDefaultFont").actual("family") == ui_theme.FAMILY
    st = ttk.Style(ui.root)
    assert st.theme_use() == "clam" and "VD.check.indicator" in st.element_names()
    b = tk.Button(ui.root, text="x")
    assert b.cget("relief") == "flat" and b.cget("bg") == ui_theme.RAISED
    assert ui_theme.hover_color("#000000") == "#1f1f1f"


def test_every_remote_command_is_documented_in_help_and_docs():
    """Help › OSC commands lists every OSC command the app accepts (from one table), and so does
    docs/REMOTE_CONTROL.md; the HTTP list covers every API path."""
    import remote_control
    import web_remote
    md = help_content.to_markdown()
    documented = [a for a, *_ in remote_control.OSC_REFERENCE]
    assert sorted(documented) == sorted(f"/visualdeck/{c}" for c in remote_control.COMMANDS)
    doc = open(os.path.join(os.path.dirname(__file__), "..", "docs", "REMOTE_CONTROL.md"), encoding="utf-8").read()
    for addr in documented:
        assert f"**{addr}  (" in md and f"`{addr}`" in doc, addr
    paths = [p for p, _ in web_remote.HTTP_REFERENCE]
    for line in web_remote.__doc__.splitlines():
        m = re.match(r"\s+(GET|POST)\s+(/\S*)", line)
        if m:
            for path in m.group(2).split("|"):
                path = path.strip()
                full = path if path.startswith("/") else "/api/" + path.lstrip("/")
                assert any(full in p for p in paths), full
    assert "Presentation Mode" in help_content.topic_titles()
    assert help_content.topic_text(dict(help_content.TOPICS)["OSC commands"]).count("/visualdeck/") >= len(documented)


def test_help_window_shows_reference_tables(blind_ci):
    """The Help window renders a reference table (term, then its text) and finds it by search."""
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    root = tk.Tk()
    root.withdraw()
    try:
        blind_ci.root = root
        blind_ci._open_help("OSC commands")
        topics, text, query = blind_ci._help_widgets
        body = text.get("1.0", "end")
        assert "/visualdeck/start_from  (cue, seconds)" in body and "Example: /visualdeck/go" in body
        query.set("display_effect")
        names = [topics.get(i).strip() for i in range(topics.size())]
        assert "OSC commands" in names and "Getting started" not in names
        query.set("by itself, like a slideshow")
        assert [topics.get(i).strip() for i in range(topics.size())] == ["Presentation Mode"]
    finally:
        root.destroy()


def test_every_menu_item_and_window_is_in_help():
    """The 1.0 scope says every window and menu item is described in Help: each
    menu label and window title in the app's code appears in the help text
    (menu items hidden by default and placeholders excepted)."""
    import inspect
    import control_interface
    import editor
    src = inspect.getsource(control_interface) + inspect.getsource(editor)
    norm = lambda x: x.replace("…", "").replace("...", "").strip()  # noqa: E731
    text = norm(" ".join(t + " " + help_content.topic_text(b) for t, b in help_content.TOPICS))
    hidden = {"Optimize", "Remove and DELETE FROM DISK", "(No recent projects)"}
    labels = [l for q, l in re.findall(r"add_(?:command|checkbutton|cascade|radiobutton)\(\s*label\s*=\s*(f?['\"])(.+?)\1",
                                       src) if not q.startswith("f")]
    missing = [l for l in labels if l not in hidden and norm(l) not in text]
    titles = [t for f, t in re.findall(r"\.title\(\s*(f?)['\"](.+?)['\"]", src) if not f]
    missing += [f"window {t}" for t in titles if norm(t) not in text]
    assert len(labels) > 50 and len(titles) > 15
    assert not missing, missing
    assert help_content.topic_titles()[0] == "Quick start: running a show"


def test_rebuilding_the_grid_keeps_its_scroll_position(ui):
    """Undo, Redo and other edits rebuild the grid; it jumped back to cue 1
    (owner's PC, acceptance run 2026-11-14)."""
    import control_interface as ci_mod
    ci = ui
    ci._build_grid = ci_mod.ControlInterface._build_grid.__get__(ci)   # the fixture stubs it out
    ci._update_cell_preview = lambda r, c: None
    ci.layer_setting_widgets = {}
    ci.grid_frame = tk.Frame(ci.root, width=500, height=300)
    ci.grid_frame.pack()
    ci._build_grid()
    ci.root.update()
    ci.grid_canvas.xview_moveto(0.4)
    ci.root.update()
    before = ci.grid_canvas.xview()[0]
    assert before > 0.2
    ci._build_grid()
    ci.root.update()
    assert abs(ci.grid_canvas.xview()[0] - before) < 0.02


def test_a_smaller_grid_leaves_no_old_cells_behind(ui, caplog):
    """Owner's PC, 2026-11-14: a grid refresh reached a cell of an earlier, larger grid
    ("bad window path name"). Rebuilding forgets the old cells; a refresh of a cell that
    is not there does nothing; an error in a Tk callback goes to the log."""
    import logging
    import control_interface as ci_mod
    ci = ui
    ci._build_grid = ci_mod.ControlInterface._build_grid.__get__(ci)
    ci.layer_setting_widgets = {}
    ci.grid_frame = tk.Frame(ci.root, width=500, height=300)
    ci.grid_frame.pack()
    cols = ci.COLS
    ci.COLS = cols + 3
    ci._build_grid()
    ci.COLS = cols
    ci._build_grid()
    assert all(c < cols for (_, c) in ci.frames_by_position) and len(ci._cell_frames) == ci.ROWS * cols
    ci.COLS = cols + 3                                   # more cues before the grid is built again
    for c in range(ci.COLS):
        ci_mod.ControlInterface._update_cell_preview(ci, 0, c)    # no error
    ci.root.report_callback_exception = ci._tk_callback_error
    with caplog.at_level(logging.ERROR, logger="visualdeck.control"):
        ci.root.after(0, lambda: 1 / 0)
        ci.root.update()
    assert any("Error in the window code" in r.getMessage() and r.exc_info for r in caplog.records)


def test_escape_closes_output_health(ui):
    """Like the other windows (owner's PC, 2026-11-14: Escape did nothing there)."""
    import control_interface as ci_mod
    ci = ui
    ci._fill_health_window = lambda *a, **k: None
    ci_mod.ControlInterface._open_health_window(ci)
    win = ci._health_win
    assert win is not None and win.winfo_exists()
    assert any('Escape' in b for b in win.bind())               # bound on the window itself
    win.update()
    win.focus_force()
    win.update()
    win.event_generate('<Escape>', when='now')
    ci.root.update()
    assert ci._health_win is None and not win.winfo_exists()


def _widgets(w):
    out = [w]
    for c in w.winfo_children():
        out += _widgets(c)
    return out


def _save_and_apply(ci):
    import control_interface as ci_mod
    ci_mod.ControlInterface._open_preferences(ci)
    win = [w for w in ci.root.winfo_children() if isinstance(w, tk.Toplevel)][-1]
    win.update()
    return win, next(b for b in _widgets(win) if isinstance(b, tk.Button) and b.cget("text") == "Save and Apply")


def test_remote_settings_leave_the_output_and_the_show_alone(ui, monkeypatch):
    """Turning on the browser remote during a show restarted the render engine
    (the output redrew and the live cue was triggered again) and marked the show
    changed (owner's PC, 2026-11-15). Only render settings do that now."""
    ci = ui
    calls = []
    for name in ("_initialize_render_engine", "_update_active_players", "_trigger_column", "_set_dirty_flag",
                 "_restart_web", "_restart_remote", "_restart_audio", "_save_config"):
        monkeypatch.setattr(ci, name, lambda *a, n=name, **k: calls.append(n))
    ci.web_enabled = False
    win, save = _save_and_apply(ci)
    web = next(w for w in _widgets(win) if w.winfo_class() == "TCheckbutton"
               and str(w.cget("text")).startswith("Serve a browser remote"))
    web.invoke()
    save.invoke()
    assert ci.web_enabled is True
    assert "_restart_web" in calls and "_save_config" in calls
    assert not {"_initialize_render_engine", "_trigger_column", "_set_dirty_flag"} & set(calls), calls
    calls.clear()
    mode = "pygame_gpu" if ci.render_mode != "pygame_gpu" else "opengl"
    win, save = _save_and_apply(ci)
    next(w for w in _widgets(win) if w.winfo_class() == "TRadiobutton" and str(w.cget("value")) == mode).invoke()
    save.invoke()
    assert ci.render_mode == mode
    assert {"_initialize_render_engine", "_trigger_column", "_set_dirty_flag"} <= set(calls), calls


def test_menu_check_marks_are_visible_without_hovering():
    """Owner's report (2026-11-15): the check mark of "GO After the Last Cue Returns to
    the First" showed only while hovered. Tk draws a menu entry's mark in its
    selectColor (the system's black on Windows) unless the entry is active."""
    if not os.environ.get("DISPLAY") and os.name != "nt":
        pytest.skip("needs a display")
    import ui_theme
    root = tk.Tk()
    try:
        ui_theme.apply(root)
        menubar = tk.Menu(root)
        sm = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Show", menu=sm)
        var = tk.BooleanVar(value=True)
        sm.add_checkbutton(label="GO After the Last Cue Returns to the First", variable=var)

        def lum(color):
            r, g, b = (c / 65535 for c in root.winfo_rgb(color))
            return 0.2126 * r + 0.7152 * g + 0.0722 * b
        mark, bg = sm.cget("selectcolor"), sm.cget("background")
        assert abs(lum(mark) - lum(bg)) > 0.5, (mark, bg)
        assert sm.entrycget(0, "selectcolor") in ("", mark)      # the entry uses the menu's color
    finally:
        root.destroy()


def test_preferences_show_one_page_at_a_time_and_save_every_page(ui, monkeypatch):
    """Preferences: pages listed on the left, the chosen page on the right
    (owner's request, 2026-11-15). Save and Apply saves the hidden pages too."""
    import control_interface as ci_mod
    ci = ui
    for name in ("_initialize_render_engine", "_update_active_players", "_trigger_column", "_set_dirty_flag",
                 "_restart_web", "_restart_remote", "_restart_audio", "_save_config"):
        monkeypatch.setattr(ci, name, lambda *a, **k: None)
    ci._prefs_page = None
    win, save = _save_and_apply(ci)
    nav = next(w for w in _widgets(win) if w.winfo_class() == "Treeview")
    assert nav.get_children() == ci_mod.ControlInterface.PREFS_PAGES
    assert nav.selection() == ("Editing the live cue",)

    def visible(text):
        w = next(w for w in _widgets(win) if "text" in w.keys() and str(w.cget("text")).startswith(text))
        return bool(w.winfo_ismapped())
    win.update()
    size = (win.winfo_width(), win.winfo_height())
    assert visible("Wait for Update Output") and not visible("Serve a browser remote")
    nav.focus("Editing the live cue")
    press(nav, "<Down>")                                     # the arrow keys move through the pages
    assert nav.selection() == ("Remote control",) and ci._prefs_page == "Remote control"
    assert visible("Serve a browser remote") and not visible("Wait for Update Output")
    nav.selection_set("Rendering")
    win.update()
    assert visible("Performance Mode") and not visible("Serve a browser remote")
    assert (win.winfo_width(), win.winfo_height()) == size     # the window keeps its size
    # A setting on a page that is not shown is saved too.
    delay = next(w for w in _widgets(win) if w.winfo_class() == "TEntry" and w.get() == str(ci.audio_delay_ms))
    delay.delete(0, "end")
    delay.insert(0, "40")
    save.invoke()
    assert ci.audio_delay_ms == 40
    win, _ = _save_and_apply(ci)                               # opens again at the page last shown
    nav = next(w for w in _widgets(win) if w.winfo_class() == "Treeview")
    assert nav.selection() == ("Rendering",)
    win.destroy()


def test_output_health_buttons_stay_put_when_warnings_change(ui):
    """A warning line that comes and goes (the render target going 60 -> 45 -> 60)
    resized the window, so the Close button jumped away under the mouse (owner's PC,
    2026-11-16). The window keeps its size once filled; the clip table gives way."""
    import control_interface as ci_mod
    ci = ui
    lines = ["effects on the output need Performance Mode"]
    ci._fill_health_window = lambda status, reasons, body, tree: reasons.config(text="\n".join(lines))
    ci_mod.ControlInterface._open_health_window(ci)
    win = ci._health_win
    win.update()
    close = [w for w in _widgets(win) if isinstance(w, (tk.Button,)) and w.cget('text') == 'Close']
    assert close, "no Close button"
    close = close[0]

    def where():
        win.update()
        return close.winfo_rooty() - win.winfo_rooty(), win.winfo_height()

    first = where()
    lines[:] = ["6 late frames in the last 5 s", "render target lowered to 45 fps under load",
                "1 cut(s) waited the full preroll time", "effects on the output need Performance Mode"]
    win.after(600, lambda: None)
    for _ in range(8):
        win.update()
        win.after(100)
    assert where() == first
    lines[:] = []
    for _ in range(8):
        win.update()
        win.after(100)
    assert where() == first
    win.event_generate('<Escape>', when='now')
    ci.root.update()
