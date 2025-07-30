"""
Performance tests for post-processors and pipeline components.

These tests verify that the system can handle large content blocks and
maintain reasonable performance characteristics under load.
"""

import gc
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

# Handle optional dependency
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from src.config import RunConfig
from src.pipeline import Pipeline
from src.post_processors import (
    EnsureBlankLineProcessor,
    OrderQuoteAnnotationProcessor,
    PostProcessorChain,
    RemoveBlankLinesInListProcessor,
    RemoveTrailingWhitespaceProcessor,
    RemoveXmlTagsProcessor,
)


class TestPerformancePostProcessors:
    """Performance tests for post-processors with large content."""

    @pytest.fixture
    def large_text_block(self):
        """Generate a large text block for performance testing."""
        lines = []
        for i in range(10000):  # 10K lines
            if i % 100 == 0:
                lines.append(f"# Header {i // 100}")
            elif i % 50 == 0:
                lines.append(f"> **Quote:** This is quote number {i // 50}. **End quote.**")
            elif i % 30 == 0:
                lines.append(f"> **Annotation:** This is annotation {i // 30}. **End annotation.**")
            elif i % 20 == 0:
                lines.append(f"* List item {i // 20}")
            elif i % 10 == 0:
                lines.append("")  # Blank line
            else:
                lines.append(f"Regular text line {i} with some content to make it realistic.")
        return "\n".join(lines)

    @pytest.fixture
    def large_xml_block(self):
        """Generate a large block with XML tags for performance testing."""
        lines = []
        for i in range(5000):  # 5K lines with XML
            lines.append(f"<p>Paragraph {i} with <strong>bold</strong> and <em>italic</em> text.</p>")
            if i % 10 == 0:
                lines.append("<br>")
            if i % 100 == 0:
                lines.append(f'<div class="section">Section {i // 100}</div>')
        return "\n".join(lines)

    def test_ensure_blank_line_performance_large_content(self, large_text_block):
        """Test EnsureBlankLineProcessor performance with large content."""
        processor = EnsureBlankLineProcessor()

        start_time = time.time()
        result = processor.process(original_block="", llm_block=large_text_block)
        end_time = time.time()

        processing_time = end_time - start_time
        lines_per_second = len(large_text_block.split("\n")) / processing_time

        # Should process at least 1000 lines per second
        assert lines_per_second > 1000, f"Performance too slow: {lines_per_second:.2f} lines/sec"
        assert len(result) > 0, "Result should not be empty"
        assert processing_time < 30, f"Processing took too long: {processing_time:.2f} seconds"

    def test_remove_xml_tags_performance_large_content(self, large_xml_block):
        """Test RemoveXmlTagsProcessor performance with large XML content."""
        processor = RemoveXmlTagsProcessor()

        start_time = time.time()
        result = processor.process(original_block="", llm_block=large_xml_block)
        end_time = time.time()

        processing_time = end_time - start_time
        chars_per_second = len(large_xml_block) / processing_time

        # Should process at least 100K characters per second
        assert chars_per_second > 100000, f"Performance too slow: {chars_per_second:.2f} chars/sec"
        assert "<p>" not in result, "XML tags should be removed"
        assert "<br>" in result, "BR tags should be preserved"
        assert processing_time < 10, f"Processing took too long: {processing_time:.2f} seconds"

    def test_order_quote_annotation_performance_large_content(self):
        """Test OrderQuoteAnnotationProcessor performance with many quote/annotation blocks."""
        processor = OrderQuoteAnnotationProcessor()

        # Generate content with many quote/annotation pairs
        blocks = []
        for i in range(1000):  # 1K quote/annotation pairs
            blocks.append(f"Regular text {i}")
            blocks.append(f"> **Annotation:** Annotation {i}. **End annotation.**")
            blocks.append(f"> **Quote:** Quote {i}. **End quote.**")
            if i % 10 == 0:
                blocks.append("")  # Blank line to separate groups

        large_block = "\n".join(blocks)

        start_time = time.time()
        result = processor.process(original_block="", llm_block=large_block)
        end_time = time.time()

        processing_time = end_time - start_time
        blocks_per_second = len(blocks) / processing_time

        # Should process at least 5000 blocks per second
        assert blocks_per_second > 5000, f"Performance too slow: {blocks_per_second:.2f} blocks/sec"
        assert len(result) > 0, "Result should not be empty"
        assert processing_time < 5, f"Processing took too long: {processing_time:.2f} seconds"

    def test_processor_chain_performance_large_content(self, large_text_block):
        """Test PostProcessorChain performance with multiple processors and large content."""
        chain = PostProcessorChain()
        chain.add_processor(RemoveXmlTagsProcessor())
        chain.add_processor(RemoveTrailingWhitespaceProcessor())
        chain.add_processor(EnsureBlankLineProcessor())
        chain.add_processor(RemoveBlankLinesInListProcessor())

        # Add some XML tags to the content
        modified_block = large_text_block.replace("Regular text", "<p>Regular text</p>")

        start_time = time.time()
        result = chain.process(original_block="", llm_block=modified_block)
        end_time = time.time()

        processing_time = end_time - start_time
        lines_per_second = len(modified_block.split("\n")) / processing_time

        # Chain should still process at least 500 lines per second
        assert lines_per_second > 500, f"Chain performance too slow: {lines_per_second:.2f} lines/sec"
        assert len(result) > 0, "Result should not be empty"
        assert processing_time < 60, f"Chain processing took too long: {processing_time:.2f} seconds"

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_usage_large_content_processing(self, large_text_block):
        """Test that memory usage doesn't grow unbounded with large content."""
        processor = EnsureBlankLineProcessor()

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process large content multiple times
        for i in range(5):
            result = processor.process(original_block="", llm_block=large_text_block)
            del result  # Explicitly delete to help garbage collection

        # Force garbage collection
        gc.collect()

        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (less than 100MB for this test)
        assert memory_growth < 100, f"Memory usage grew too much: {memory_growth:.2f} MB"

    def test_very_long_lines_performance(self):
        """Test performance with very long individual lines."""
        # Create content with very long lines (10K characters each)
        long_lines = []
        for i in range(100):
            line = f"Line {i}: " + "x" * 10000
            long_lines.append(line)

        long_line_block = "\n".join(long_lines)
        processor = RemoveTrailingWhitespaceProcessor()

        start_time = time.time()
        processor.process(original_block="", llm_block=long_line_block)
        end_time = time.time()

        processing_time = end_time - start_time
        chars_per_second = len(long_line_block) / processing_time

        # Should handle very long lines efficiently
        assert chars_per_second > 1000000, f"Long line performance too slow: {chars_per_second:.2f} chars/sec"
        assert processing_time < 5, f"Long line processing took too long: {processing_time:.2f} seconds"


class TestPerformancePipeline:
    """Performance tests for pipeline operations."""

    def test_file_copying_performance_large_files(self):
        """Test file copying performance with large files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a large input file (10MB)
            large_content = "Line of text with some content.\n" * 300000  # ~10MB
            input_file = temp_path / "large_input.md"
            input_file.write_text(large_content, encoding="utf-8")

            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Create mock config
            mock_config = Mock(spec=RunConfig)
            mock_config.input_file = input_file
            mock_config.output_dir = output_dir
            mock_config.book_name = "test_book"
            mock_config.author_name = "test_author"
            mock_config.original_file = input_file
            mock_config.length_reduction = [50]
            mock_config.phases = []
            mock_config.get_phase_order.return_value = []

            pipeline = Pipeline(config=mock_config)

            start_time = time.time()
            pipeline._copy_input_file_to_output()
            end_time = time.time()

            copying_time = end_time - start_time
            file_size_mb = len(large_content) / 1024 / 1024
            mb_per_second = file_size_mb / copying_time

            # Should copy at least 50 MB/second
            assert mb_per_second > 50, f"File copying too slow: {mb_per_second:.2f} MB/sec"
            assert copying_time < 1, f"File copying took too long: {copying_time:.2f} seconds"

            # Verify file was copied correctly
            expected_output_file = output_dir / "00-large_input.md"
            assert expected_output_file.exists()
            assert expected_output_file.stat().st_size == input_file.stat().st_size

    def test_concurrent_processor_safety(self):
        """Test that processors are safe for concurrent use."""
        import queue
        import threading

        # Generate test content
        large_text_block = "Regular text line.\n" * 1000  # 1K lines

        processor = EnsureBlankLineProcessor()
        results = queue.Queue()
        exceptions = queue.Queue()

        def process_block(block_id):
            try:
                # Add unique content to each block
                modified_block = large_text_block.replace("Regular text", f"Regular text {block_id}")
                result = processor.process(original_block="", llm_block=modified_block)
                results.put((block_id, len(result)))
            except Exception as e:
                exceptions.put((block_id, e))

        # Start multiple threads processing simultaneously
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_block, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
            assert not thread.is_alive(), "Thread didn't complete in time"

        # Check that no exceptions occurred
        assert exceptions.empty(), f"Exceptions occurred: {list(exceptions.queue)}"

        # Check that all results were produced
        assert results.qsize() == 5, f"Expected 5 results, got {results.qsize()}"

        # Verify all results are reasonable
        while not results.empty():
            block_id, result_length = results.get()
            assert result_length > 0, f"Block {block_id} produced empty result"


class TestMemoryUsage:
    """Memory usage pattern tests."""

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_cleanup_after_processing(self):
        """Test that memory is properly cleaned up after processing."""
        processor = EnsureBlankLineProcessor()

        # Generate a large block
        large_block = "Line of text.\n" * 50000  # ~50K lines

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process and immediately delete results
        for i in range(10):
            result = processor.process(original_block="", llm_block=large_block)
            del result

            # Periodic garbage collection
            if i % 3 == 0:
                gc.collect()

        # Final garbage collection
        gc.collect()

        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Memory should not have grown significantly (less than 50MB)
        assert memory_growth < 50, f"Memory leak detected: {memory_growth:.2f} MB growth"

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_peak_memory_usage_large_processing(self):
        """Test peak memory usage during large content processing."""
        processor_chain = PostProcessorChain()
        processor_chain.add_processor(RemoveXmlTagsProcessor())
        processor_chain.add_processor(EnsureBlankLineProcessor())

        # Create very large content block
        huge_block = "<p>Content line with various tags.</p>\n" * 100000  # ~100K lines

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process the huge block
        result = processor_chain.process(original_block="", llm_block=huge_block)

        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - initial_memory

        # Clean up
        del result
        del huge_block
        gc.collect()

        # Peak memory usage should be reasonable (less than 500MB for this test)
        assert memory_usage < 500, f"Peak memory usage too high: {memory_usage:.2f} MB"
