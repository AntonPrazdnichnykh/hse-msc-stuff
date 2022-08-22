import hunspell
from n_grams import BiGramModel
from textdistance import levenshtein, jaro_winkler
from metaphone import doublemetaphone
from typing import Tuple
from collections.abc import Callable
import re
from nltk import pos_tag


class SpellChecker:
    def __init__(
            self,
            url1: str = 'https://www.norvig.com/ngrams/count_1w.txt',
            url2: str = 'https://www.norvig.com/ngrams/count_2w.txt',
            weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),  # levenshtein, jaro-winkler, phonetic, neg logprob
    ):
        self.spellchecker = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
        self.bigram_model = BiGramModel(url1, url2)
        self.weights = weights

    def _weight_avg(self, x):
        n = len(x)
        assert n == len(self.weights)
        return sum(x_ * w for x_, w in zip(x, self.weights)) / n

    @staticmethod
    def _phonetic_sim(w1, w2):
        w1_enc1, w1_enc2 = doublemetaphone(w1)
        w2_enc1, w2_enc2 = doublemetaphone(w2)
        if w1_enc1 == w2_enc1:
            return 2.
        if w1_enc1 == w2_enc2 or w1_enc2 == w2_enc1:
            return 1.
        if w1_enc2 == w2_enc2:
            return 0.5
        return 0.


    def get_fearures(self, triplet, candidates):
        prev, wrong, foll = triplet
        features = [
            (
                levenshtein(wrong, cand),
                1 - jaro_winkler(wrong, cand),
                -self._phonetic_sim(wrong, cand),
                self.bigram_model.nll_bi([cand, foll], sos=prev)
            ) for cand in candidates
        ]
        return features

    def _range(self, features):
        return min(range(len(features)), key=lambda idx: self._weight_avg(features[idx]))

    def add_to_dict(self, words):
        for w in words:
            self.spellchecker.add(w)

    def correct_words(self, text):
        words_pos = [(m.start(), m.end()) for m in re.finditer(r'\w+', text)]
        words_tagged = pos_tag(re.findall(r'\w+', text))
        corrected = []
        words_tagged = [('<S>', 'SOS')] + words_tagged + [('<E>', 'EOS')]
        for i in range(1, len(words_tagged) - 1):
            word, tag = words_tagged[i]
            if not word.isalpha() or tag == 'NNP' or self.spellchecker.spell(word):
                corrected.append(word)
            else:
                prev = corrected[-1] if corrected else words_tagged[-1][0]
                foll, _ = words_tagged[i + 1]
                suggestions = self.spellchecker.suggest(word)
                # print(suggestions, self.get_fearures((prev, word, foll), suggestions))
                best_idx = self._range(self.get_fearures((prev, word, foll), suggestions))
                corrected.append(suggestions[best_idx] if suggestions else word)
        text_corrected = ''
        last_idx = 0
        for (start, end), word in zip(words_pos, corrected):
            text_corrected += (text[last_idx: start] + word)
            last_idx = end
        return text_corrected


if __name__ == "__main__":
    spellchecker = SpellChecker()
    text = "Zis tekst contains som erors."
    print(spellchecker.correct_words(text))
