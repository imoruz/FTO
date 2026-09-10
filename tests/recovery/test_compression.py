import pytest

from fto.recovery import compression
from fto.recovery.compression import (
    LLMLINGUA2_MODEL,
    LONGLLMLINGUA_MODEL,
    CompressionResult,
    ContextCompressor,
    LLMLinguaCompressor,
)


class FakeTokenizer:
    """Token count as whitespace-separated words; enough to check the arithmetic."""

    def encode(self, text):
        return text.split()


class FakePromptCompressor:
    """Stand-in for llmlingua's PromptCompressor.

    Compresses by keeping the first half of the words of each context, and
    records every call so tests can assert on the parameters fto passes.
    """

    instances = []

    def __init__(self, model_name, use_llmlingua2, device_map):
        self.model_name = model_name
        self.use_llmlingua2 = use_llmlingua2
        self.device_map = device_map
        self.oai_tokenizer = FakeTokenizer()
        self.calls = []
        FakePromptCompressor.instances.append(self)

    def compress_prompt(self, context, **kwargs):
        self.calls.append((context, kwargs))
        compressed = [' '.join(c.split()[: max(1, len(c.split()) // 2)]) for c in context]
        return {
            'compressed_prompt': '\n\n'.join(compressed),
            'compressed_prompt_list': compressed,
            'origin_tokens': 100,
            'compressed_tokens': 50,
        }


@pytest.fixture
def fake_llmlingua(fake_module, monkeypatch):
    """Serve fto's lazy `from llmlingua import PromptCompressor` from a fake."""
    monkeypatch.setattr(compression, '_MODEL_CACHE', {})
    FakePromptCompressor.instances = []
    fake_module('llmlingua', PromptCompressor=FakePromptCompressor)
    return FakePromptCompressor


class TestCompressionResult:
    def test_rate_is_compressed_over_origin(self):
        result = CompressionResult(['a'], origin_tokens=200, compressed_tokens=50)
        assert result.rate == 0.25

    def test_rate_is_one_when_nothing_was_compressed(self):
        assert CompressionResult(['a']).rate == 1.0

    def test_describe_reports_both_token_counts(self):
        result = CompressionResult([], origin_tokens=200, compressed_tokens=50)
        assert result.describe() == '200 -> 50 tokens (25.0% of original)'


class TestContextCompressorBase:
    def test_compress_passes_the_contexts_through(self):
        result = ContextCompressor().compress(['one', 'two'])
        assert result.texts == ['one', 'two']

    def test_compress_returns_a_copy(self):
        contexts = ['one']
        assert ContextCompressor().compress(contexts).texts is not contexts


class TestLLMLinguaCompressorDefaults:
    def test_defaults_to_the_llmlingua2_model(self):
        compressor = LLMLinguaCompressor()
        assert compressor.use_llmlingua2 is True
        assert compressor.model_name == LLMLINGUA2_MODEL

    def test_falls_back_to_the_causal_model_for_longllmlingua(self):
        assert LLMLinguaCompressor(use_llmlingua2=False).model_name == LONGLLMLINGUA_MODEL

    def test_an_explicit_model_name_wins(self):
        assert LLMLinguaCompressor(model_name='my/model').model_name == 'my/model'

    def test_constructing_it_does_not_load_a_model(self):
        # The scoring model is hundreds of MB; building the config must be free.
        LLMLinguaCompressor()
        assert not compression._MODEL_CACHE


class TestLLMLingua2Compression:
    def test_compresses_every_entry_and_keeps_them_aligned(self, fake_llmlingua):
        compressor = LLMLinguaCompressor()

        result = compressor.compress(['a b c d', 'e f g h'])

        assert result.texts == ['a b', 'e f']

    def test_reports_token_counts_from_the_texts_it_handled(self, fake_llmlingua):
        result = LLMLinguaCompressor().compress(['a b c d', 'e f g h'])

        assert result.origin_tokens == 8
        assert result.compressed_tokens == 4
        assert result.rate == 0.5

    def test_entries_without_text_are_passed_through_untouched(self, fake_llmlingua):
        result = LLMLinguaCompressor().compress(['a b c d', '', '   ', 'e f g h'])

        assert result.texts == ['a b', '', '   ', 'e f']
        # The model is only shown the entries that had something to compress.
        context, _ = fake_llmlingua.instances[0].calls[0]
        assert context == ['a b c d', 'e f g h']

    def test_nothing_to_compress_never_loads_the_model(self, fake_llmlingua):
        result = LLMLinguaCompressor().compress(['', '  '])

        assert result.texts == ['', '  ']
        assert fake_llmlingua.instances == []

    def test_context_level_filtering_is_off_so_entries_cannot_be_dropped(
        self, fake_llmlingua
    ):
        LLMLinguaCompressor().compress(['a b c d'])

        _, kwargs = fake_llmlingua.instances[0].calls[0]
        assert kwargs['use_context_level_filter'] is False

    def test_forwards_the_compression_settings(self, fake_llmlingua):
        compressor = LLMLinguaCompressor(
            rate=0.3,
            target_token=512,
            force_tokens=['TASK_COMPLETE'],
            force_reserve_digit=False,
            drop_consecutive=True,
            params={'token_to_word': 'max'},
        )

        compressor.compress(['a b c d'])

        _, kwargs = fake_llmlingua.instances[0].calls[0]
        assert kwargs['rate'] == 0.3
        assert kwargs['target_token'] == 512
        assert kwargs['force_tokens'] == ['TASK_COMPLETE']
        assert kwargs['force_reserve_digit'] is False
        assert kwargs['drop_consecutive'] is True
        assert kwargs['token_to_word'] == 'max'

    def test_a_misaligned_result_is_an_error(self, fake_llmlingua, monkeypatch):
        monkeypatch.setattr(
            FakePromptCompressor,
            'compress_prompt',
            lambda self, context, **kwargs: {
                'compressed_prompt_list': ['only one'],
                'origin_tokens': 1,
                'compressed_tokens': 1,
            },
        )

        with pytest.raises(RuntimeError, match='cannot map them back'):
            LLMLinguaCompressor().compress(['a b', 'c d'])

    def test_the_model_is_loaded_once_and_reused(self, fake_llmlingua):
        compressor = LLMLinguaCompressor()

        compressor.compress(['a b c d'])
        compressor.compress(['e f g h'])
        LLMLinguaCompressor().compress(['i j k l'])

        assert len(fake_llmlingua.instances) == 1

    def test_a_different_configuration_loads_its_own_model(self, fake_llmlingua):
        LLMLinguaCompressor().compress(['a b c d'])
        LLMLinguaCompressor(model_name='other/model').compress(['a b c d'])

        assert len(fake_llmlingua.instances) == 2

    def test_the_device_is_resolved_when_the_model_loads(self, fake_llmlingua):
        LLMLinguaCompressor(device_map='cpu').compress(['a b c d'])

        assert fake_llmlingua.instances[0].device_map == 'cpu'


class TestLongLLMLinguaCompression:
    def test_compresses_one_entry_at_a_time_to_keep_them_separable(
        self, fake_llmlingua
    ):
        compressor = LLMLinguaCompressor(use_llmlingua2=False)

        result = compressor.compress(['a b c d', 'e f g h'], question='what now')

        assert result.texts == ['a b', 'e f']
        contexts = [context for context, _ in fake_llmlingua.instances[0].calls]
        assert contexts == [['a b c d'], ['e f g h']]

    def test_conditions_on_the_question(self, fake_llmlingua):
        LLMLinguaCompressor(use_llmlingua2=False).compress(['a b c d'], question='review this')

        _, kwargs = fake_llmlingua.instances[0].calls[0]
        assert kwargs['question'] == 'review this'
        assert kwargs['rank_method'] == 'longllmlingua'
        assert kwargs['condition_in_question'] == 'after_condition'
        assert kwargs['condition_compare'] is True
        # The question is a message of its own; it must not be glued on again.
        assert kwargs['concate_question'] is False

    def test_falls_back_to_plain_llmlingua_without_a_question(self, fake_llmlingua):
        LLMLinguaCompressor(use_llmlingua2=False).compress(['a b c d'])

        _, kwargs = fake_llmlingua.instances[0].calls[0]
        assert kwargs['question'] == ''
        assert kwargs['rank_method'] == 'llmlingua'
        assert kwargs['condition_in_question'] == 'none'
        assert kwargs['condition_compare'] is False

    def test_explicit_params_override_the_defaults(self, fake_llmlingua):
        compressor = LLMLinguaCompressor(
            use_llmlingua2=False, params={'reorder_context': 'original'}
        )

        compressor.compress(['a b c d'], question='q')

        _, kwargs = fake_llmlingua.instances[0].calls[0]
        assert kwargs['reorder_context'] == 'original'
