"""Tests for book_writer generator module."""

from book_writer.generator import (
    NO_CHANGES_MARKER,
    _feedback_indicates_no_changes,
    get_chapter_filename,
)
from book_writer.models import ChapterOutline


class TestFeedbackIndicatesNoChanges:
    """Tests for _feedback_indicates_no_changes detection logic."""

    def test_no_changes_recommended_only(self):
        """Test feedback with only 'No Changes Recommended' returns True."""
        feedback = """### No Changes Recommended

The passage is well-crafted. No refinements are needed."""
        assert _feedback_indicates_no_changes(feedback) is True

    def test_changes_present_returns_false(self):
        """Test feedback with change markers returns False."""
        feedback = """### Change [1]: Clarity

**Location:** First paragraph

**Current text:** "The thing is complex."

**Proposed change:** "The concept is complex."

**Rationale:** More precise language."""
        assert _feedback_indicates_no_changes(feedback) is False

    def test_no_changes_marker_with_change_markers_returns_false(self):
        """Test the edge case ChatGPT identified: 'no changes' for one aspect but changes for others."""
        feedback = """### Change [1]: Clarity

**Location:** First paragraph

**Current text:** "The thing is complex."

**Proposed change:** "The concept is complex."

**Rationale:** More precise language.

### Change [2]: Voice

**Location:** Second paragraph

**Current text:** "It was done."

**Proposed change:** "We accomplished this."

**Rationale:** More active voice.

The accuracy of the text is good as is. No Changes Recommended for accuracy."""
        assert _feedback_indicates_no_changes(feedback) is False

    def test_partial_no_changes_mention_with_changes(self):
        """Test another variation where model mentions no changes for one category."""
        feedback = """### Change [1]: Clarity - improve terminology

**Location:** Opening sentence

**Current text:** "Interest rates do stuff."

**Proposed change:** "Interest rates influence economic activity."

**Rationale:** More professional language.

Note: The voice and flow are excellent. No Changes Recommended for those aspects."""
        assert _feedback_indicates_no_changes(feedback) is False

    def test_empty_feedback(self):
        """Test empty feedback returns False (no marker present)."""
        assert _feedback_indicates_no_changes("") is False

    def test_feedback_without_any_markers(self):
        """Test feedback without any recognized markers returns False."""
        feedback = "The text looks good overall. Some minor suggestions..."
        assert _feedback_indicates_no_changes(feedback) is False

    def test_case_sensitivity_of_marker(self):
        """Test that marker matching is case-sensitive."""
        # The exact marker should work
        feedback = f"### {NO_CHANGES_MARKER}\n\nThe passage is fine."
        assert _feedback_indicates_no_changes(feedback) is True

        # Different case should not match
        feedback_lower = "### no changes recommended\n\nThe passage is fine."
        assert _feedback_indicates_no_changes(feedback_lower) is False

    def test_marker_anywhere_in_text(self):
        """Test that marker is detected anywhere in the feedback."""
        feedback = f"""## Analysis

After careful review, I conclude:

### {NO_CHANGES_MARKER}

This section is well-written."""
        assert _feedback_indicates_no_changes(feedback) is True


class TestGetChapterFilename:
    """Tests for get_chapter_filename helper."""

    def test_regular_chapter_single_digit(self):
        """Test regular chapter with single digit."""
        chapter = ChapterOutline(id="1", number=1, title="Chapter One")
        assert get_chapter_filename(chapter) == "chapter_01.md"

    def test_regular_chapter_double_digit(self):
        """Test regular chapter with double digit."""
        chapter = ChapterOutline(id="12", number=12, title="Chapter Twelve")
        assert get_chapter_filename(chapter) == "chapter_12.md"

    def test_regular_chapter_no_number(self):
        """Test chapter filename when number is missing."""
        chapter = ChapterOutline(id="3", title="Chapter Three")
        assert get_chapter_filename(chapter) == "chapter_03.md"
