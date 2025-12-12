from books.base_builder import BaseBookBuilder


def test_make_blockquote_lists_loose_inserts_blank_quote_lines_between_items() -> None:
    input_md = (
        "> **Key Terms/Concepts**\n> *Tyranny of the majority* - A\n> *Social tyranny* - B\n> *Harm principle* - C\n"
    )
    expected_md = (
        "> **Key Terms/Concepts**\n"
        ">\n"
        "> *Tyranny of the majority* - A\n"
        ">\n"
        "> *Social tyranny* - B\n"
        ">\n"
        "> *Harm principle* - C\n"
    )

    assert BaseBookBuilder.make_blockquote_lists_loose(input_md) == expected_md


def test_make_blockquote_lists_loose_does_not_double_insert_when_already_loose() -> None:
    input_md = "> **Key Terms/Concepts**\n>\n> *Tyranny of the majority* - A\n>\n> *Social tyranny* - B\n"
    assert BaseBookBuilder.make_blockquote_lists_loose(input_md) == input_md


def test_make_blockquote_lists_loose_does_not_change_non_blockquote_lists() -> None:
    input_md = "* A\n* B\n"
    assert BaseBookBuilder.make_blockquote_lists_loose(input_md) == input_md
