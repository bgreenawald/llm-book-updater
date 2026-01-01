"""Tests for book_writer models module."""

from datetime import datetime

from book_writer.models import (
    BookConfig,
    BookOutline,
    BookState,
    ChapterOutline,
    ChapterState,
    ChapterStatus,
    GenerationConfig,
    PhaseModels,
    SectionOutline,
    SectionState,
    SectionStatus,
)


class TestSectionStatus:
    """Tests for SectionStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all expected statuses exist."""
        assert SectionStatus.PENDING == "pending"
        assert SectionStatus.IN_PROGRESS == "in_progress"
        assert SectionStatus.COMPLETED == "completed"
        assert SectionStatus.FAILED == "failed"


class TestChapterStatus:
    """Tests for ChapterStatus enum."""

    def test_all_statuses_exist(self):
        """Test that all expected statuses exist."""
        assert ChapterStatus.PENDING == "pending"
        assert ChapterStatus.IN_PROGRESS == "in_progress"
        assert ChapterStatus.COMPLETED == "completed"
        assert ChapterStatus.PARTIAL == "partial"
        assert ChapterStatus.FAILED == "failed"


class TestSectionOutline:
    """Tests for SectionOutline model."""

    def test_create_section_outline(self):
        """Test creating a section outline."""
        section = SectionOutline(
            id="1.1",
            title="Introduction",
            outline_content="This section introduces...",
        )
        assert section.id == "1.1"
        assert section.title == "Introduction"
        assert section.heading_level == 2
        assert section.line_start == 0
        assert section.line_end == 0

    def test_custom_heading_level(self):
        """Test section with custom heading level."""
        section = SectionOutline(
            id="1.1.1",
            title="Sub-section",
            heading_level=3,
            outline_content="Content",
        )
        assert section.heading_level == 3


class TestSectionState:
    """Tests for SectionState model."""

    def test_default_values(self):
        """Test default values for section state."""
        state = SectionState(section_id="1.1")
        assert state.section_id == "1.1"
        assert state.status == SectionStatus.PENDING
        assert state.retry_count == 0
        assert state.last_error is None
        assert state.generated_content is None
        assert state.started_at is None
        assert state.completed_at is None
        assert state.token_count is None

    def test_completed_state(self):
        """Test a completed section state."""
        now = datetime.now()
        state = SectionState(
            section_id="1.1",
            status=SectionStatus.COMPLETED,
            generated_content="Generated text",
            started_at=now,
            completed_at=now,
            token_count=500,
        )
        assert state.status == SectionStatus.COMPLETED
        assert state.generated_content == "Generated text"
        assert state.token_count == 500


class TestChapterOutline:
    """Tests for ChapterOutline model."""

    def test_create_chapter_outline(self):
        """Test creating a chapter outline."""
        chapter = ChapterOutline(
            id="1",
            number=1,
            title="Introduction",
        )
        assert chapter.id == "1"
        assert chapter.number == 1
        assert chapter.title == "Introduction"
        assert chapter.goals is None
        assert chapter.sections == []
        assert chapter.summary_box is None

    def test_chapter_with_sections(self):
        """Test chapter with sections."""
        sections = [
            SectionOutline(id="1.1", title="First", outline_content="Content 1"),
            SectionOutline(id="1.2", title="Second", outline_content="Content 2"),
        ]
        chapter = ChapterOutline(
            id="1",
            title="Chapter One",
            sections=sections,
        )
        assert len(chapter.sections) == 2
        assert chapter.sections[0].id == "1.1"
        assert chapter.sections[1].id == "1.2"


class TestChapterState:
    """Tests for ChapterState model."""

    def test_default_values(self):
        """Test default values for chapter state."""
        state = ChapterState(chapter_id="1")
        assert state.chapter_id == "1"
        assert state.status == ChapterStatus.PENDING
        assert state.sections == {}
        assert state.started_at is None
        assert state.completed_at is None

    def test_chapter_with_section_states(self):
        """Test chapter with section states."""
        section_states = {
            "1.1": SectionState(section_id="1.1"),
            "1.2": SectionState(section_id="1.2", status=SectionStatus.COMPLETED),
        }
        state = ChapterState(
            chapter_id="1",
            status=ChapterStatus.IN_PROGRESS,
            sections=section_states,
        )
        assert len(state.sections) == 2
        assert state.sections["1.1"].status == SectionStatus.PENDING
        assert state.sections["1.2"].status == SectionStatus.COMPLETED


class TestBookOutline:
    """Tests for BookOutline model."""

    def test_minimal_book_outline(self):
        """Test creating minimal book outline."""
        outline = BookOutline(title="Test Book")
        assert outline.title == "Test Book"
        assert outline.preface is None
        assert outline.parts == []
        assert outline.chapters == []
        assert outline.appendices == []
        assert outline.final_notes is None

    def test_complete_book_outline(self):
        """Test creating complete book outline."""
        preface = ChapterOutline(id="preface", title="Preface")
        chapter = ChapterOutline(id="1", number=1, title="Chapter 1")
        appendix = ChapterOutline(id="appendix_a", title="Appendix A")

        outline = BookOutline(
            title="Complete Book",
            preface=preface,
            parts=["Part I: Foundations", "Part II: Advanced"],
            chapters=[chapter],
            appendices=[appendix],
            final_notes="Some notes",
        )
        assert outline.preface is not None
        assert len(outline.parts) == 2
        assert len(outline.chapters) == 1
        assert len(outline.appendices) == 1
        assert outline.final_notes == "Some notes"


class TestBookState:
    """Tests for BookState model."""

    def test_create_book_state(self):
        """Test creating book state."""
        now = datetime.now()
        state = BookState(
            rubric_hash="abc123",
            model="test/model",
            created_at=now,
            updated_at=now,
        )
        assert state.rubric_hash == "abc123"
        assert state.model == "test/model"
        assert state.chapters == {}

    def test_get_pending_sections(self):
        """Test getting pending sections."""
        now = datetime.now()
        state = BookState(
            rubric_hash="hash",
            model="model",
            created_at=now,
            updated_at=now,
            chapters={
                "1": ChapterState(
                    chapter_id="1",
                    sections={
                        "1.1": SectionState(section_id="1.1", status=SectionStatus.PENDING),
                        "1.2": SectionState(section_id="1.2", status=SectionStatus.COMPLETED),
                        "1.3": SectionState(section_id="1.3", status=SectionStatus.FAILED),
                    },
                ),
                "2": ChapterState(
                    chapter_id="2",
                    sections={
                        "2.1": SectionState(section_id="2.1", status=SectionStatus.PENDING),
                    },
                ),
            },
        )
        pending = state.get_pending_sections()
        # Should include pending and failed sections
        assert ("1", "1.1") in pending
        assert ("1", "1.3") in pending
        assert ("2", "2.1") in pending
        assert ("1", "1.2") not in pending
        assert len(pending) == 3

    def test_get_completed_sections(self):
        """Test getting completed sections for a chapter."""
        now = datetime.now()
        state = BookState(
            rubric_hash="hash",
            model="model",
            created_at=now,
            updated_at=now,
            chapters={
                "1": ChapterState(
                    chapter_id="1",
                    sections={
                        "1.1": SectionState(
                            section_id="1.1",
                            status=SectionStatus.COMPLETED,
                            generated_content="Content 1",
                        ),
                        "1.2": SectionState(section_id="1.2", status=SectionStatus.PENDING),
                        "1.3": SectionState(
                            section_id="1.3",
                            status=SectionStatus.COMPLETED,
                            generated_content="Content 3",
                        ),
                    },
                ),
            },
        )
        completed = state.get_completed_sections("1")
        assert len(completed) == 2
        section_ids = [s[0] for s in completed]
        assert "1.1" in section_ids
        assert "1.3" in section_ids

    def test_get_completed_sections_nonexistent_chapter(self):
        """Test getting completed sections for nonexistent chapter."""
        now = datetime.now()
        state = BookState(
            rubric_hash="hash",
            model="model",
            created_at=now,
            updated_at=now,
        )
        completed = state.get_completed_sections("nonexistent")
        assert completed == []


class TestBookConfig:
    """Tests for BookConfig model."""

    def test_default_values(self):
        """Test default values."""
        config = BookConfig()
        assert config.title == "Untitled Book"
        assert config.model == "anthropic/claude-sonnet-4"
        assert config.phase_models is None
        assert config.max_concurrent_chapters == 5

    def test_custom_values(self):
        """Test custom values."""
        phase_models = PhaseModels(
            generate="anthropic/claude-sonnet-4",
            identify="anthropic/claude-haiku-4",
            implement="anthropic/claude-sonnet-4",
        )
        config = BookConfig(
            title="My Book",
            model="openai/gpt-4",
            phase_models=phase_models,
            max_concurrent_chapters=10,
        )
        assert config.title == "My Book"
        assert config.model == "openai/gpt-4"
        assert config.phase_models == phase_models
        assert config.max_concurrent_chapters == 10


class TestPhaseModels:
    """Tests for PhaseModels model."""

    def test_default_values(self):
        """Test default values."""
        config = PhaseModels()
        assert config.generate is None
        assert config.identify is None
        assert config.implement is None

    def test_custom_values(self):
        """Test custom values."""
        config = PhaseModels(
            generate="anthropic/claude-sonnet-4",
            identify="anthropic/claude-haiku-4",
            implement="anthropic/claude-opus-4",
        )
        assert config.generate == "anthropic/claude-sonnet-4"
        assert config.identify == "anthropic/claude-haiku-4"
        assert config.implement == "anthropic/claude-opus-4"


class TestGenerationConfig:
    """Tests for GenerationConfig model."""

    def test_default_values(self):
        """Test default values."""
        config = GenerationConfig()
        assert config.model == "anthropic/claude-sonnet-4"
        assert config.phase_models == PhaseModels()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.max_concurrent_chapters == 5

    def test_custom_values(self):
        """Test custom values."""
        config = GenerationConfig(
            model="test/model",
            phase_models=PhaseModels(
                generate="phase1/model",
                identify="phase2/model",
                implement="phase3/model",
            ),
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            max_concurrent_chapters=3,
        )
        assert config.model == "test/model"
        assert config.phase_models.generate == "phase1/model"
        assert config.phase_models.identify == "phase2/model"
        assert config.phase_models.implement == "phase3/model"
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.max_concurrent_chapters == 3

    def test_get_model_for_phase(self):
        """Test phase-specific model selection."""
        config = GenerationConfig(
            model="default/model",
            phase_models=PhaseModels(
                generate="phase1/model",
                identify=None,
                implement="phase3/model",
            ),
        )
        assert config.get_model_for_phase(1) == "phase1/model"
        assert config.get_model_for_phase(2) == "default/model"
        assert config.get_model_for_phase(3) == "phase3/model"
