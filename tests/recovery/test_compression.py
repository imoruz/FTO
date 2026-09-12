import pytest

from fto.recovery import compression
from fto.recovery.compression import (
    DEFAULT_FORCE_TOKENS,
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


def make_compressor(**kwargs):
    """A compressor whose short-fragment threshold is out of the way.

    These tests work with a handful of words per entry; `min_fragment_chars`
    has its own class below.
    """
    kwargs.setdefault('min_fragment_chars', 0)
    return LLMLinguaCompressor(**kwargs)


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
        compressor = make_compressor()

        result = compressor.compress(['a b c d', 'e f g h'])

        assert result.texts == ['a b', 'e f']

    def test_reports_token_counts_from_the_texts_it_handled(self, fake_llmlingua):
        result = make_compressor().compress(['a b c d', 'e f g h'])

        assert result.origin_tokens == 8
        assert result.compressed_tokens == 4
        assert result.rate == 0.5

    def test_entries_without_text_are_passed_through_untouched(self, fake_llmlingua):
        result = make_compressor().compress(['a b c d', '', '   ', 'e f g h'])

        assert result.texts == ['a b', '', '   ', 'e f']
        # The model is only shown the entries that had something to compress.
        context, _ = fake_llmlingua.instances[0].calls[0]
        assert context == ['a b c d', 'e f g h']

    def test_nothing_to_compress_never_loads_the_model(self, fake_llmlingua):
        result = make_compressor().compress(['', '  '])

        assert result.texts == ['', '  ']
        assert fake_llmlingua.instances == []

    def test_context_level_filtering_is_off_so_entries_cannot_be_dropped(
        self, fake_llmlingua
    ):
        make_compressor().compress(['a b c d'])

        _, kwargs = fake_llmlingua.instances[0].calls[0]
        assert kwargs['use_context_level_filter'] is False

    def test_forwards_the_compression_settings(self, fake_llmlingua):
        compressor = make_compressor(
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
            make_compressor().compress(['a b', 'c d'])

    def test_the_model_is_loaded_once_and_reused(self, fake_llmlingua):
        compressor = make_compressor()

        compressor.compress(['a b c d'])
        compressor.compress(['e f g h'])
        make_compressor().compress(['i j k l'])

        assert len(fake_llmlingua.instances) == 1

    def test_a_different_configuration_loads_its_own_model(self, fake_llmlingua):
        make_compressor().compress(['a b c d'])
        make_compressor(model_name='other/model').compress(['a b c d'])

        assert len(fake_llmlingua.instances) == 2

    def test_the_device_is_resolved_when_the_model_loads(self, fake_llmlingua):
        make_compressor(device_map='cpu').compress(['a b c d'])

        assert fake_llmlingua.instances[0].device_map == 'cpu'


class TestLongLLMLinguaCompression:
    def test_compresses_one_entry_at_a_time_to_keep_them_separable(
        self, fake_llmlingua
    ):
        compressor = make_compressor(use_llmlingua2=False)

        result = compressor.compress(['a b c d', 'e f g h'], question='what now')

        assert result.texts == ['a b', 'e f']
        contexts = [context for context, _ in fake_llmlingua.instances[0].calls]
        assert contexts == [['a b c d'], ['e f g h']]

    def test_conditions_on_the_question(self, fake_llmlingua):
        make_compressor(use_llmlingua2=False).compress(['a b c d'], question='review this')

        _, kwargs = fake_llmlingua.instances[0].calls[0]
        assert kwargs['question'] == 'review this'
        assert kwargs['rank_method'] == 'longllmlingua'
        assert kwargs['condition_in_question'] == 'after_condition'
        assert kwargs['condition_compare'] is True
        # The question is a message of its own; it must not be glued on again.
        assert kwargs['concate_question'] is False

    def test_falls_back_to_plain_llmlingua_without_a_question(self, fake_llmlingua):
        make_compressor(use_llmlingua2=False).compress(['a b c d'])

        _, kwargs = fake_llmlingua.instances[0].calls[0]
        assert kwargs['question'] == ''
        assert kwargs['rank_method'] == 'llmlingua'
        assert kwargs['condition_in_question'] == 'none'
        assert kwargs['condition_compare'] is False

    def test_explicit_params_override_the_defaults(self, fake_llmlingua):
        compressor = make_compressor(
            use_llmlingua2=False, params={'reorder_context': 'original'}
        )

        compressor.compress(['a b c d'], question='q')

        _, kwargs = fake_llmlingua.instances[0].calls[0]
        assert kwargs['reorder_context'] == 'original'


class TestQuestionIsModeDependent:
    """LLMLingua-2 is task-agnostic: upstream drops `question` before it can
    reach `compress_prompt_llmlingua2`. Advertising that stops a caller from
    reading a conditioning guarantee into the signature."""

    def test_llmlingua2_does_not_use_the_question(self):
        assert LLMLinguaCompressor().uses_question is False

    def test_longllmlingua_does_use_the_question(self):
        assert LLMLinguaCompressor(use_llmlingua2=False).uses_question is True

    def test_the_passthrough_base_does_not_use_it_either(self):
        assert ContextCompressor().uses_question is False

    def test_llmlingua2_is_never_handed_a_question(self, fake_llmlingua):
        make_compressor().compress(['a b c d'], question='review this')

        _, kwargs = fake_llmlingua.instances[0].calls[0]
        assert 'question' not in kwargs

    def test_a_question_passed_anyway_changes_nothing(self, fake_llmlingua):
        compressor = make_compressor()

        with_q = compressor.compress(['a b c d'], question='review this')
        without_q = compressor.compress(['a b c d'])

        assert with_q.texts == without_q.texts


class TestModelSharingAcrossRates:
    """Age-tiering means several compressors at different rates. The cache key
    leaves `rate` out, so tiering costs no extra load or memory."""

    def test_rate_is_not_part_of_the_cache_key(self, fake_llmlingua):
        make_compressor(rate=0.6).compress(['a b c d'])
        make_compressor(rate=0.35).compress(['e f g h'])

        assert len(fake_llmlingua.instances) == 1

    def test_each_rate_still_reaches_the_model(self, fake_llmlingua):
        make_compressor(rate=0.6).compress(['a b c d'])
        make_compressor(rate=0.35).compress(['e f g h'])

        rates = [kwargs['rate'] for _, kwargs in fake_llmlingua.instances[0].calls]
        assert rates == [0.6, 0.35]


class TestForceTokenDefaults:
    def test_the_full_stop_is_not_forced(self):
        # Forcing '.' makes LLMLingua-2 re-join it as its own word, turning
        # "6.4.6" into "6. 4. 6" and "cpe.go" into "cpe. go".
        assert '.' not in DEFAULT_FORCE_TOKENS

    def test_newline_and_colon_are_forced(self):
        assert '\n' in DEFAULT_FORCE_TOKENS
        assert ':' in DEFAULT_FORCE_TOKENS

    def test_negations_are_forced_in_both_cases(self):
        for word in ('not', 'no', 'none', 'never', 'only', 'must'):
            assert word in DEFAULT_FORCE_TOKENS
            assert word.capitalize() in DEFAULT_FORCE_TOKENS

    def test_the_compressor_defaults_to_them(self):
        assert LLMLinguaCompressor().force_tokens == DEFAULT_FORCE_TOKENS

    def test_the_default_list_is_not_shared_between_instances(self):
        compressor = LLMLinguaCompressor()
        compressor.force_tokens.append('MINE')
        assert 'MINE' not in LLMLinguaCompressor().force_tokens
        assert 'MINE' not in DEFAULT_FORCE_TOKENS


class TestCodeSpanProtection:
    """Pruned code is not shorter, it is wrong.

    `cpe.go` comes back as `cpe.` and "cpe:2.3:h:fortinet" as ":2.:fortinet:",
    which leaves the restarted node a path or literal that looks real and is
    not -- so code never reaches the model.
    """

    def test_a_backtick_span_survives_verbatim(self, fake_llmlingua):
        result = make_compressor().compress(['edit the file `pkg/cpe/cpe.go` now please'])

        assert '`pkg/cpe/cpe.go`' in result.texts[0]

    def test_a_fenced_block_survives_verbatim(self, fake_llmlingua):
        fenced = '```go\nreturn util.Unique(cpes)\n```'
        result = make_compressor().compress([f'apply this edit exactly:\n{fenced}\nthen re-read it'])

        assert fenced in result.texts[0]

    def test_only_the_prose_reaches_the_model(self, fake_llmlingua):
        make_compressor().compress(['keep the `cpe.go` path and the `6.4.6` version'])

        context, _ = fake_llmlingua.instances[0].calls[0]
        assert context == ['keep the ', ' path and the ', ' version']

    def test_prose_around_the_code_is_still_compressed(self, fake_llmlingua):
        result = make_compressor().compress(['one two three four `cpe.go` five six seven eight'])

        # FakePromptCompressor keeps the first half of each fragment's words.
        assert result.texts[0] == 'one two `cpe.go` five six'

    def test_reassembly_does_not_weld_words_onto_a_code_span(self, fake_llmlingua):
        # The compressor strips the whitespace around the fragment it was
        # given, so the separator has to be put back.
        result = make_compressor().compress(['read `cpe.go` twice'])

        assert result.texts[0] == 'read `cpe.go` twice'

    def test_an_entry_that_is_only_code_is_never_sent(self, fake_llmlingua):
        result = make_compressor().compress(['`cpe.go`'])

        assert result.texts == ['`cpe.go`']
        assert fake_llmlingua.instances == []

    def test_several_entries_keep_their_alignment_across_fragments(self, fake_llmlingua):
        result = make_compressor().compress([
            'alpha beta gamma delta',
            'read `a.go` and `b.go` twice over now',
            'epsilon zeta eta theta',
        ])

        assert len(result.texts) == 3
        assert '`a.go`' in result.texts[1] and '`b.go`' in result.texts[1]
        assert result.texts[0] == 'alpha beta'
        assert result.texts[2] == 'epsilon zeta'

    def test_protection_applies_to_longllmlingua_too(self, fake_llmlingua):
        # force_tokens are ignored by the v1 API; span protection is not.
        compressor = make_compressor(use_llmlingua2=False)

        result = compressor.compress(['keep `cpe.go` intact here'], question='q')

        assert '`cpe.go`' in result.texts[0]

    def test_protection_can_be_turned_off_for_an_ablation(self, fake_llmlingua):
        compressor = make_compressor(protect_code=False)

        compressor.compress(['keep the `cpe.go` path and the `6.4.6` version'])

        context, _ = fake_llmlingua.instances[0].calls[0]
        assert context == ['keep the `cpe.go` path and the `6.4.6` version']

    def test_token_counts_cover_the_protected_spans(self, fake_llmlingua):
        # The reported ratio is the reduction the node actually sees, not the
        # reduction of the prose the model was shown.
        result = make_compressor().compress(['one two three four `a.go` five six seven eight'])

        assert result.origin_tokens == 9
        assert result.compressed_tokens == 5


class TestShortFragmentThreshold:
    """Protecting code splits a message into many short prose runs.

    In a plan dense with backticks most of those runs are a few words of glue.
    Each one costs its own padded forward pass and saves almost nothing, so
    below `min_fragment_chars` they are kept verbatim.
    """

    def test_it_is_on_by_default(self):
        assert LLMLinguaCompressor().min_fragment_chars == 80

    def test_short_prose_between_code_spans_is_kept_verbatim(self, fake_llmlingua):
        result = LLMLinguaCompressor(min_fragment_chars=80).compress(
            ['read `a.go` and then `b.go` twice']
        )

        assert result.texts == ['read `a.go` and then `b.go` twice']
        assert fake_llmlingua.instances == []

    def test_long_prose_is_still_compressed(self, fake_llmlingua):
        long_run = 'word ' * 30
        result = LLMLinguaCompressor(min_fragment_chars=80).compress(
            [f'{long_run}`a.go` and then `b.go`']
        )

        context, _ = fake_llmlingua.instances[0].calls[0]
        # Only the long run went to the model; the glue between the spans did not.
        assert context == [long_run]
        assert '`a.go`' in result.texts[0] and '`b.go`' in result.texts[0]

    def test_the_threshold_measures_the_stripped_fragment(self, fake_llmlingua):
        # 40 real characters padded out with whitespace stays under a 60 limit.
        padded = '\n\n' + ('abcde ' * 8).strip() + '\n\n'
        LLMLinguaCompressor(min_fragment_chars=60).compress([f'{padded}`a.go`'])

        assert fake_llmlingua.instances == []

    def test_zero_compresses_every_fragment(self, fake_llmlingua):
        LLMLinguaCompressor(min_fragment_chars=0).compress(['read `a.go` now'])

        context, _ = fake_llmlingua.instances[0].calls[0]
        assert context == ['read ', ' now']

    def test_a_message_with_no_code_is_one_fragment_and_obeys_the_threshold(
        self, fake_llmlingua
    ):
        # The threshold reads as "prose runs shorter than this are kept
        # verbatim", and a message without code spans is a single prose run.
        # Nothing worth saving in a line that short anyway.
        compressor = LLMLinguaCompressor(min_fragment_chars=80)

        assert compressor.compress(['a b c d']).texts == ['a b c d']

        long_run = 'word ' * 30
        assert compressor.compress([long_run]).texts != [long_run]
