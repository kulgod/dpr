PASSAGE_LENGTH_WORDS = 100

class FileParser:
    def __init__(self):
        pass

    def get_passages(self, data): 
        words = data.split(' ') 
        all_passages = []
        n_words = len(words)

        for i in range(0, n_words // PASSAGE_LENGTH_WORDS):
            starting_word_index = i * PASSAGE_LENGTH_WORDS
            passage_words = words[starting_word_index: starting_word_index + PASSAGE_LENGTH_WORDS]
            all_passages.append(' '.join(passage_words))

        return all_passages