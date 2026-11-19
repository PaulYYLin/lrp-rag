import unittest
from unittest.mock import patch

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from lrp.phase4_qa import make_input, pack


class EvidenceBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vocab = {v: i for i, v in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        backend = Tokenizer(models.BPE(vocab, []))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, clean_up_tokenization_spaces=False)
        cls.tokenizer.chat_template = "{% for m in messages %}{{m['content']}}{% endfor %}"

    def test_cut_inside_multibyte_character(self):
        with patch("lrp.phase4_qa.BUDGET", 2):
            _, visible, total = pack(self.tokenizer, "a中z")
        self.assertEqual((visible, total), ("a", 5))

    def test_ranked_context_and_missing_facts(self):
        pages = {"a": dict(title="A", text="ok! more")}
        one = dict(gold_title="A", gold_sentence_index=0, status="mapped", page_id="a",
                   char_start=0, char_end=3, sentence="ok!")
        two = dict(one, gold_sentence_index=1, char_start=3, char_end=8, sentence=" more")
        missing = dict(gold_title="B", gold_sentence_index=0, status="title_missing")
        q = dict(id="q", text="?", supporting_facts=[one, one, two, missing])
        with patch("lrp.phase4_qa.BUDGET", 5):
            _, r = make_input(self.tokenizer, q, pages, ["a"], "raw768-flat")
        self.assertEqual(r["context"], "A\nok!")
        self.assertEqual(r["facts"], [["A", 0, "visible"], ["A", 1, "truncated"], ["B", 0, "unmapped"]])
        self.assertEqual((r["sentence_recall"], r["passage_recall"], r["complete"], r["fully_mapped"]),
                         (1/3, 1/2, 0, False))

    def test_one_missing_character_is_not_full_evidence(self):
        page = dict(title="A", text="abc")
        fact = dict(gold_title="A", gold_sentence_index=0, status="mapped", page_id="a",
                    char_start=0, char_end=3, sentence="abc")
        q = dict(id="q", text="?", supporting_facts=[fact])
        for budget, expected in ((4, 0), (5, 1)):
            with patch("lrp.phase4_qa.BUDGET", budget):
                _, row = make_input(self.tokenizer, q, {"a": page}, ["a"], "flat")
                self.assertEqual(row["complete"], expected)
        _, row = make_input(self.tokenizer, q, {}, [], "question_only")
        self.assertEqual((row["context_tokens"], row["facts"]), (0, [["A", 0, "not_retrieved"]]))
