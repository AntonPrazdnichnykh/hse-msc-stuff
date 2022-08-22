import requests
from collections import Counter
from math import log


def load_counts(text: str, sep: str ='\t'):
    C = Counter()
    for l in text.rstrip('\n').split('\n'):
        key, count = l.split(sep)
        C[key] = int(count)
    return C


class BiGramModel:
    _eps = 1e-8
    def __init__(
            self,
            url1: str = 'https://www.norvig.com/ngrams/count_1w.txt',
            url2: str = 'https://www.norvig.com/ngrams/count_2w.txt'
    ):
        self.counter1 = load_counts(requests.get(url1).text)
        self.counter2 = load_counts(requests.get(url2).text)
        self.n1 = sum(list(self.counter1.values()))
        self.n2 = sum(list(self.counter2.values()))

    def p1w(self, x):
        return self.counter1[x] / self.n1

    def p2w(self, x):
        return self.counter2[x] / self.n2

    def pr_c(self, word, prev):
        bigram = prev + ' ' + word
        p_bi = self.p2w(bigram)
        p_prev = self.p1w(prev)
        if p_prev:
            return p_bi / p_prev
        return self.p1w(word)

    def nll_uni(self, words):
        return -sum(log(self.p1w(w) + self._eps) for w in words) / len(words)

    def nll_bi(self, words, sos='<S>'):
        return -sum(log(self.pr_c(w, sos * (i == 0) + words[i-1] * (i > 0)) + self._eps) for i, w in enumerate(words)) / (len(words) + 1)
