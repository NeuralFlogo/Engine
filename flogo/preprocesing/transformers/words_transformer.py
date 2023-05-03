class WordsTransformer:
    def __init__(self, strings_iter, vocabulary, input_size):
        self.strings_iter = strings_iter
        self.transformer = vocabulary.build(strings_iter)
        self.num_words = round(input_size / 2)
        self.input_size = input_size
        self.inputs = []
        self.outputs = []
        self.__process_texts()

    def __process_texts(self):
        for text in self.strings_iter:
            self.__process_tokens(self.transformer(text))

    def __process_tokens(self, tokens):
        for index in range(len(tokens)):
            self.outputs.append(tokens[index])
            if index < self.num_words:
                self.inputs.append(self.__get_initial_input_words(tokens, index))
            elif index >= len(tokens) - self.num_words:
                self.inputs.append(self.__get_final_words(tokens, index))
            else:
                self.inputs.append(self.__get_input_words_from(tokens[index - self.num_words:index + self.num_words]))

    def __get_initial_input_words(self, tokens, index):
        inputs = []
        for sub_index in range(self.input_size):
            if sub_index != index:
                inputs.append(tokens[sub_index])
        return inputs

    def __get_final_words(self, tokens, index):
        inputs = []
        for sub_index in range(len(tokens) - self.input_size, len(tokens)):
            if sub_index != index:
                inputs.append(tokens[sub_index])
        return inputs

    def __get_input_words_from(self, words_list):
        inputs = []
        for index in range(len(words_list)):
            if index != self.num_words:
                inputs.append(words_list[index])
        return inputs

    def transform_inputs(self):
        return self.inputs

    def transform_outputs(self):
        return self.outputs
