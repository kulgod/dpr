import os
import nltk 
import ssl

PASSAGE_LENGTH_WORDS = 100
PARSING_ROLLING_WINDOW_NUM_WORDS = 20

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

    def get_passages_with_complete_sentences(self, data):
        sentences = nltk.sent_tokenize(data)
        all_passages = []

        curr_passage_word_count = 0
        curr_passage_sentences = []
        for s in sentences:
            sentence_words = s.split(' ')
            n_words_in_sentence = len(sentence_words)

            if curr_passage_word_count + n_words_in_sentence > PASSAGE_LENGTH_WORDS:
                # if we've hit the wordl limit, "commit" this passage and reset for the next one
                all_passages.append(' '.join(curr_passage_sentences))

                # start the next passage with the current sentence so it doesn't get lost
                curr_passage_word_count = n_words_in_sentence
                curr_passage_sentences = [s]
            else:
                curr_passage_word_count += n_words_in_sentence
                curr_passage_sentences.append(s)

        return all_passages

    def get_rolling_passages(self, data):
        words = data.split(' ') 
        all_passages = []
        n_words = len(words)

        i = 0
        while i < n_words: 
            passage_words = words[i: i + PASSAGE_LENGTH_WORDS]
            all_passages.append(' '.join(passage_words))
            i += PARSING_ROLLING_WINDOW_NUM_WORDS

        return all_passages

# test parsing with rolling windows
if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    p = FileParser()

    with open(os.path.join(root_dir, '../data/13sentences.txt'), 'r') as f:
        file_contents = f.read()
        
        for psg in p.get_passages_with_complete_sentences(file_contents)[:5]:
            print(psg)
            print('-'*25)
