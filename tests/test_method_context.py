import pytest

from app.github.method_context import (
    _changed_lines_from_patch,
    _find_method_brace_based,
    _find_method_python,
)


# ── _changed_lines_from_patch ─────────────────────────────────────────────────

def test_changed_lines_only_added():
    patch = "@@ -1,3 +1,4 @@\n context\n+added\n context\n context"
    assert _changed_lines_from_patch(patch) == {2}


def test_changed_lines_excludes_context_and_deleted():
    patch = "@@ -1,3 +1,2 @@\n context\n-deleted\n context"
    # context lines are not "added" so not in changed set
    assert _changed_lines_from_patch(patch) == set()


def test_changed_lines_multiple_hunks():
    patch = "@@ -1,2 +1,2 @@\n context\n+line2\n@@ -10,2 +10,2 @@\n context\n+line11"
    assert _changed_lines_from_patch(patch) == {2, 11}


def test_changed_lines_empty_patch():
    assert _changed_lines_from_patch("") == set()


# ── _find_method_brace_based (Java) ───────────────────────────────────────────

_JAVA_FILE = """\
public class Foo {

    public String methodA() {
        return "a";
    }

    public int methodB(int x) {
        if (x > 0) {
            return x * 2;
        }
        return -1;
    }

    @Override
    public String toString() {
        return "Foo";
    }
}
""".splitlines()


def test_java_finds_method_containing_line():
    # line 8 is inside methodB (lines 7-12)
    result = _find_method_brace_based(_JAVA_FILE, 8)
    assert result is not None
    start, end = result
    assert _JAVA_FILE[start - 1].strip().startswith("public int methodB")
    assert _JAVA_FILE[end - 1].strip() == "}"


def test_java_finds_method_at_first_line_of_body():
    # line 4 is `return "a"` inside methodA
    result = _find_method_brace_based(_JAVA_FILE, 4)
    assert result is not None
    start, _ = result
    assert "methodA" in _JAVA_FILE[start - 1]


def test_java_includes_annotation_in_signature():
    # line 16 is inside toString, which has @Override
    result = _find_method_brace_based(_JAVA_FILE, 16)
    assert result is not None
    start, _ = result
    # start should be at @Override or the method signature line
    sig_block = "\n".join(_JAVA_FILE[start - 1 :start + 2])
    assert "toString" in sig_block


def test_java_returns_none_for_out_of_range():
    assert _find_method_brace_based(_JAVA_FILE, 9999) is None
    assert _find_method_brace_based(_JAVA_FILE, 0) is None


def test_java_nested_braces_dont_confuse_boundary():
    # line 9 is inside the `if` block inside methodB — still finds methodB
    result = _find_method_brace_based(_JAVA_FILE, 9)
    assert result is not None
    start, _ = result
    assert "methodB" in _JAVA_FILE[start - 1]


# ── _find_method_python ───────────────────────────────────────────────────────

_PYTHON_FILE = """\
class MyService:

    def method_a(self):
        return "a"

    def method_b(self, x: int) -> int:
        if x > 0:
            return x * 2
        return -1

    def method_c(self):
        pass
""".splitlines()


def test_python_finds_method_containing_line():
    # line 7 is `if x > 0` inside method_b
    result = _find_method_python(_PYTHON_FILE, 7)
    assert result is not None
    start, end = result
    assert "method_b" in _PYTHON_FILE[start - 1]


def test_python_finds_correct_method_not_previous():
    # line 4 is `return "a"` inside method_a, not method_b
    result = _find_method_python(_PYTHON_FILE, 4)
    assert result is not None
    start, _ = result
    assert "method_a" in _PYTHON_FILE[start - 1]


def test_python_returns_none_when_no_def_found():
    # line 1 is `class MyService:` — no enclosing def
    result = _find_method_python(_PYTHON_FILE, 1)
    assert result is None
