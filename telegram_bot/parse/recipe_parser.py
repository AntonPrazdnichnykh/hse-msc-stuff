from natasha import Segmenter, Doc, NewsMorphTagger, NewsEmbedding, MorphVocab
from parse.command_parser import CommandParser
import urllib
import urllib.parse
import urllib.request
import urllib.error
import re


class RecipeParser(CommandParser):
    __except_nouns = [
        'бот',
        'рецепт',
        'блюдо',
        'привет'
    ]
    __segmenter = Segmenter()
    __emb = NewsEmbedding()
    __morph_tagger = NewsMorphTagger(__emb)
    __morph_vocab = MorphVocab()

    def __init__(self):
        self.keywords = ["приготовить", "сделать", "из", "рецепт", "блюдо"]

    def _extract_ingredients(self, message: Doc):
        message.segment(self.__segmenter)
        message.tag_morph(self.__morph_tagger)
        ingredients = []
        for token in message.tokens:
            if token.pos == 'NOUN' and token.text.lower() not in self.__except_nouns:
                token.lemmatize(self.__morph_vocab)
                ingredients.append(token.lemma)
        return ingredients

    def process(self, query):
        # query = self._extract_ingredients(message)
        url = "https://www.povarenok.ru/recipes/search/?ing="
        for word in query:
            url += (urllib.parse.quote_plus(word.encode('cp1251')) + '%2C+')
        url = url[:-1] + "&ing_exc=&kitchen=&type=&cat=&subcat=&orderby=#searchformtop"

        try:
            all_found_recepies = urllib.request.urlopen(url).read().decode('cp1251')
        except urllib.error.HTTPError:
            return None

        try:
            first_found = re.search(r'https\:\/\/www\.povarenok\.ru\/recipes\/show\/\d+\/', all_found_recepies).group(0)
        except AttributeError:
            return None

        recipe = urllib.request.urlopen(first_found).read().decode('cp1251')

        out = {}

        res = re.search(r'<title>([\w\s;&\-]+) – кулинарный рецепт', recipe)
        out['name'] = res.group(1) if res is not None else 'Интересный рецепт'

        # print(out['name'])

        ingrid_amounts = [''.join(x) for x in re.findall(
            r'<span itemprop="name">([\w\s]+)<\/span>|<span itemprop="amount"(>[\d\.\s\w]+)</span>', recipe)]

        ingredients = []

        for item in ingrid_amounts:
            if item[0] == '>':
                ingredients[-1] += f' -- {item[1:]}'
            else:
                ingredients.append(item)

        out['ingredients'] = ingredients
        try:
            recipe_raw = re.search(r'<h2>Рецепт([\s\S]+?)<div class="article-tags', recipe).group(0)

            recipe_steps = re.findall(r'<p>([\s\S]+?)<\/p>', recipe_raw)

            recipe_steps = [re.sub(r'<br \/>', '', step) for step in recipe_steps]

            out['steps'] = recipe_steps

            return out
        except Exception as e:
            return None


if __name__ == '__main__':

    parser = RecipeParser()

    text = 'Что приготовить из томатов'
    doc = Doc(text)
    print(parser.process(doc))
