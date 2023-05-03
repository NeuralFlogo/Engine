from torchtext.vocab import build_vocab_from_iterator


class PytorchVocabulary:

    def __init__(self, tokenizer, min_word_tokenizer):
        self.tokenizer = tokenizer
        self.min_word_tokenizer = min_word_tokenizer

    def build(self, data_iter):
        vocab = build_vocab_from_iterator(
            map(self.tokenizer, data_iter),
            specials=["<unk>"],
            min_freq=self.min_word_tokenizer,
        )
        vocab.set_default_index(vocab["<unk>"])
        return self.__get_transformer(vocab)

    def __get_transformer(self, vocab):
        return lambda x: vocab(self.tokenizer(x))

