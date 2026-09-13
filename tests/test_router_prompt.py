import pytest
from crimson0020.translation_runtime.contracts import (
    TermConstraint,
    TermSuggestion,
    TranslationBatchRequest,
    TranslationRequestItem,
)
from crimson0020.translation_runtime.router.prompt import (
    PromptBuilder,
    RouterPromptEnvelope,
    MAX_TOTAL_CHARS,
)
from crimson0020.translation_runtime.router.response_parser import RouterResponseParser

def test_prompt_builder_structure_and_bounds():
    builder = PromptBuilder()
    items = [
        TranslationRequestItem(
            work_id="item_1",
            masked_text="Hello world",
            context=("scene: start",),
            context_signature="sig1",
            hard_terms=(TermConstraint(source_term="world", target_term="thế giới"),),
            soft_terms=(),
        ),
        TranslationRequestItem(
            work_id="item_2",
            masked_text="Welcome to the game",
            context=(),
            context_signature="sig2",
            hard_terms=(),
            soft_terms=(TermSuggestion(source_term="game", suggested_target="trò chơi"),),
        ),
    ]
    batch = TranslationBatchRequest(
        job_id="job1",
        source_language="en",
        target_language="vi",
        items=tuple(items),
    )
    envelope = builder.build_batch_prompt(batch)
    assert isinstance(envelope, RouterPromptEnvelope)
    assert envelope.system_instruction != ""
    assert "item_1" in envelope.user_prompt
    assert "item_2" in envelope.user_prompt
    assert "thế giới" in envelope.user_prompt
    assert "trò chơi" in envelope.user_prompt

    # Must enforce 48 items limit
    too_many = [
        TranslationRequestItem(
            work_id=f"item_{i}",
            masked_text="test",
            context=(),
            context_signature="sig",
            hard_terms=(),
            soft_terms=(),
        )
        for i in range(50)
    ]
    with pytest.raises(ValueError):
        builder.build_prompt(items=too_many)

    # Must enforce characters limit
    oversized = [
        TranslationRequestItem(
            work_id="item_big",
            masked_text="A" * (MAX_TOTAL_CHARS + 100),
            context=(),
            context_signature="sig",
            hard_terms=(),
            soft_terms=(),
        )
    ]
    with pytest.raises(ValueError):
        builder.build_prompt(items=oversized)

def test_response_parser_valid_json_and_markdown_fence():
    parser = RouterResponseParser()
    raw_response = """```json
    {
      "translations": [
        {"id": "item_1", "text": "Xin chào thế giới"},
        {"id": "item_2", "text": "Chào mừng đến với trò chơi"}
      ]
    }
    ```"""
    res = parser.parse_response(raw_response, expected_ids=["item_1", "item_2"])
    assert res.is_success
    assert res.translations["item_1"] == "Xin chào thế giới"
    assert res.translations["item_2"] == "Chào mừng đến với trò chơi"

def test_response_parser_array_variant():
    parser = RouterResponseParser()
    raw_response = """[
        {"id": "item_1", "text": "Xin chào thế giới"},
        {"id": "item_2", "text": "Chào mừng đến với trò chơi"}
    ]"""
    res = parser.parse_response(raw_response, expected_ids=["item_1", "item_2"])
    assert res.is_success
    assert res.translations["item_1"] == "Xin chào thế giới"
    assert res.translations["item_2"] == "Chào mừng đến với trò chơi"

def test_response_parser_partial_and_ambiguous_ids():
    parser = RouterResponseParser()
    # item_1 is valid, item_3 is unknown, item_2 is missing
    raw_response = """{
        "translations": [
            {"id": "item_1", "text": "Xin chào"},
            {"id": "item_3", "text": "Lạ lùng"}
        ]
    }"""
    res = parser.parse_response(raw_response, expected_ids=["item_1", "item_2"])
    assert "item_1" in res.translations
    assert "item_2" in res.missing_ids
    assert "item_3" in res.rejected_ids

def test_response_parser_duplicate_ids_rejected():
    parser = RouterResponseParser()
    raw_response = """{
        "translations": [
            {"id": "item_1", "text": "Bản dịch 1"},
            {"id": "item_1", "text": "Bản dịch 2"}
        ]
    }"""
    res = parser.parse_response(raw_response, expected_ids=["item_1"])
    # Duplicate IDs must be rejected for ambiguity
    assert "item_1" in res.rejected_ids
    assert "item_1" not in res.translations

def test_response_parser_empty_and_corrupt():
    parser = RouterResponseParser()
    empty_res = parser.parse_response("", expected_ids=["item_1"])
    assert not empty_res.is_success
    assert "item_1" in empty_res.missing_ids

    corrupt_res = parser.parse_response("Sorry I cannot translate that.", expected_ids=["item_1"])
    assert not corrupt_res.is_success
    assert "item_1" in corrupt_res.missing_ids

def test_response_parser_dict_map():
    parser = RouterResponseParser()
    raw = '{"item_1": "Xin chào", "item_2": "Thế giới"}'
    res = parser.parse_response(raw, expected_ids=["item_1", "item_2"])
    assert res.is_success
    assert res.translations["item_1"] == "Xin chào"
    assert res.translations["item_2"] == "Thế giới"
