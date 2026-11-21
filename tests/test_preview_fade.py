# SPDX-License-Identifier: MPL-2.0
"""The output preview follows fades and shows the cue a fade ends on."""
from test_blind_editing import ci as blind_ci  # noqa: F401
from test_cue_activation import ci as _base_ci  # noqa: F401


def test_previews_are_requested_during_a_fade_at_a_limited_rate(blind_ci):
    ci = blind_ci
    ci.is_video_playing = False
    ci.preview_dirty = False
    assert not ci._preview_due(False, 10.0, 9.0), "nothing changed, nothing to do"
    assert ci._preview_due(True, 10.0, 9.0), "a fade always wants a preview"
    ci.preview_dirty = True
    assert not ci._preview_due(True, 10.03, 10.0), "but at most about 15 per second"
    assert ci._preview_due(True, 10.07, 10.0)
    ci.is_video_playing = True
    assert not ci._preview_due(False, 10.02, 10.0) and ci._preview_due(False, 10.04, 10.0)


def test_the_end_of_a_fade_asks_for_a_preview(blind_ci):
    ci = blind_ci
    asked = []
    ci.set_preview_dirty = lambda: asked.append(ci.active_col)    # the fixture stubs it out; record instead
    ci._initiate_transition(0, 1)
    asked.clear()
    ci._finalize_transition_from_thread(1)
    assert asked, "a still cue reached by a fade must be shown"
    ci._blackout()
    asked.clear()
    ci._finalize_transition_from_thread(None)
    assert asked, "and so must black"
