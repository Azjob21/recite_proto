import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from engine import normalize, tokenize, simulate, RevealTracker

REF = [n for _, n in tokenize("بسم الله الرحمن الرحيم الحمد لله رب العالمين")]
T = 0.85

def test_normalize_unifies_variants():
    assert normalize("ٱللَّهُ") == normalize("الله") == "الله"
    assert normalize("أَحَدٌ") == "احد"

def test_waqf_marks_glued_to_previous_word():
    w = tokenize("قُلْ هُوَ ۖ ٱللَّهُ")
    assert [n for _, n in w] == ["قل", "هو", "الله"] and "ۖ" in w[1][0]

def test_all_correct():
    r = simulate(REF, 0, "بسم الله الرحمن الرحيم".split(), T)
    assert r.cursor == 4 and r.error is None

def test_wrong_word_detected():
    r = simulate(REF, 0, "بسم الله الحمد الرحيم".split(), T)
    assert r.error and r.error["index"] == 2 and r.error["heard"] == "الحمد"
    r = simulate(REF, 0, "بسم الله الرحيم الرحمن".split(), T)   # swapped: heard next word
    assert r.error and r.error["index"] == 2 and r.error["kind"] == "skip"

def test_skipped_word_detected():
    r = simulate(REF, 0, "بسم الرحمن".split(), T)
    assert r.error and r.error["kind"] == "skip" and r.error["index"] == 1

def test_stutter_tolerated():
    r = simulate(REF, 0, "بسم الله الله الرحمن".split(), T)
    assert r.cursor == 3 and r.error is None

def test_asr_split_and_merge_tolerated():
    r = simulate(REF, 4, "الحمد لله رب العالمين".split(), T)
    assert r.cursor == 8 and r.error is None
    r = simulate(REF, 0, "بسماله الرحمن".split(), T)      # merged "بسم الله"
    assert r.cursor == 3 and r.error is None

def test_reread_from_start_anchors_back():
    r = simulate(REF, 3, "بسم الله الرحمن الرحيم".split(), T)
    assert r.cursor == 4 and r.error is None

def test_interim_needs_agreement_and_lag():
    t = RevealTracker(REF, thr=T)
    t.update("بسم الله".split(), False, 1)               # last word dropped, no prev
    assert t.revealed == 0
    t.update("بسم الله الرحمن".split(), False, 1)        # agreement -> 1 word stable
    assert t.revealed == 1
    t.update("بسم الله الرحمن الرحيم".split(), False, 1)
    assert t.revealed == 2
    t.update("بسم الله الرحمن الرحيم".split(), True, 1)  # final commits everything
    assert t.revealed == 4

def test_interim_error_needs_two_confirmations():
    t = RevealTracker(REF, thr=T)
    t.update("بسم الله الرحيم الحمد".split(), False, 1)
    assert t.state()["error"] is None
    assert t.update("بسم الله الرحيم الحمد لله".split(), False, 1) is True
    s = t.state()
    assert s["error"]["index"] == 2 and s["revealed"] == 2

def test_error_clears_after_correct_reread():
    t = RevealTracker(REF, thr=T)
    t.update("بسم الله الرحيم".split(), True, 1)
    assert t.state()["error"]
    t.update("الرحمن الرحيم".split(), True, 2)
    s = t.state()
    assert s["error"] is None and s["revealed"] == 4
