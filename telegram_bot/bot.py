import telebot
import config
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger
from deeppavlov import configs, build_model
from parse.recipe_parser import RecipeParser
from parse.user import User

bot = telebot.TeleBot(config.BOT_TOKEN)

tokenizer_tox = AutoTokenizer.from_pretrained("sismetanin/rubert-toxic-pikabu-2ch")
toxic_model = AutoModelForSequenceClassification.from_pretrained("sismetanin/rubert-toxic-pikabu-2ch")
ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)

user_dict = {}

recipe_parser = RecipeParser()

segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()


@bot.message_handler(commands=['start'])  # Функция отвечает на команду 'start'
def start_message(message):
    bot.send_message(message.chat.id,
                 f"Привет!\n"
                 f"Введи список ингредиентов, и я подскажу тебе, что можно из них приготовить️🍝\n\n"
                 f"Чтобы узнать полный список команд, напиши /help \n"
                 f"Чтобы закончить диалог, напиши /exit\n",
                 parse_mode='HTML')


@bot.message_handler(commands=['help'])  # Функция отвечает на команду 'help'
def help_message(message):
    bot.send_message(message.chat.id,
                 f"<b>Я знаю следующие команды</b>:\n\n"
                 f"/help - <i>Повторить это сообщение</i>\n\n"
                 f"/exit - <i>Выход</i>\n\n"
                 f"Если хочешь узнать, что можно приготовить из данных ингредиентов -- напиши, я тебя пойму!",
                 parse_mode='HTML')


@bot.message_handler(commands=['exit'])  # Функция отвечает на команду 'exit'
def end_message(message):
    user_dict[message.chat.id].needs_greet = True
    bot.send_message(message.chat.id,
                     f"Рад был помочь! До встречи!\n")


def is_toxic(message):
    message = Doc(message.text)
    message.segment(segmenter)
    message.tag_morph(morph_tagger)
    for token in message.tokens:
        token.lemmatize(morph_vocab)
    tokens_pt = tokenizer_tox(message.text, return_tensors="pt")
    with torch.no_grad():
        pred = torch.nn.functional.softmax(toxic_model(**tokens_pt)[0], dim=1).squeeze()
        request_is_toxic = pred[1] > 0.8
        return request_is_toxic


def is_appology(message):
    message = Doc(message.text)
    message.segment(segmenter)
    message.tag_morph(morph_tagger)
    cnt = 0
    for token in message.tokens:
        token.lemmatize(morph_vocab)
        if token.lemma.lower() in ['извинить', 'извинение', 'простить', 'прощение', 'извини']:
            cnt += 1
    return cnt


def is_bye(message):
    message = Doc(message.text)
    message.segment(segmenter)
    message.tag_morph(morph_tagger)
    cnt = 0
    for token in message.tokens:
        token.lemmatize(morph_vocab)
        if token.lemma.lower() in ['пока', 'свидание']:
            cnt += 1
    return cnt


@bot.message_handler(content_types=['text'])  # Функция обрабатывает текстовые сообщения
def get_text(message):
    chat_id = message.chat.id
    tox = is_toxic(message)
    if tox:
        if chat_id in user_dict:
            user_dict[chat_id].toxic = tox
        else:
            user_dict[chat_id] = User(tox)
        bot.reply_to(message,
                    f"Ну нет, так дела не делаются.\n\n"
                    f"Не буду вам помогать, пока не извинитесь."
                    )
        # bot.register_next_step_handler(msg, get_text)
        return

    if chat_id in user_dict and user_dict[chat_id].toxic:
        if is_appology(message):
            bot.reply_to(message, "Ваши извинения приняты! Теперь мои бесконечные знания рецептов снова в вашем "
                                  "распоряжении!")
            user_dict[chat_id].toxic = 0
        else:
            bot.reply_to(message, "Я по-прежнему жду ваших извинений...")
        # bot.register_next_step_handler(msg, get_text)
        return
    if chat_id not in user_dict:
        user_dict[chat_id] = User(tox)
    if is_bye(message):
        user_dict[chat_id].needs_greet = True
        bot.send_message(chat_id, "Рад была помочь! До встречи!")
        return
    if user_dict[chat_id].needs_greet:
        user_dict[chat_id].needs_greet = False
        bot.send_message(chat_id, "Приветики-пистолетики!")
        return

    # query classification
    recipe_cnt = 0
    doc = Doc(message.text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        if token.lemma in recipe_parser.keywords:
            recipe_cnt += 1

    if recipe_cnt == 0:
        bot.reply_to(message, 'Кажется, я не понял, что вы от меня хотите((((\n'
                              'Я пока умею только рецепты подсказывать.')
        return
    bot.reply_to(message, 'Могу предложить такой вариантик:')
    process_recipe_step(message)


def process_recipe_step(message):
    chat_id = message.chat.id
    recipe_ingredients = recipe_parser._extract_ingredients(Doc(message.text))
    if len(recipe_ingredients) == 0:
        bot.reply_to(message, 'Я не знаю таких ингридиентов(((\nСпросите меня рецепт из чего-нибудь другого.')
        # bot.register_next_step_handler(msg, get_text)
    else:
        recipe = recipe_parser.process(recipe_ingredients)
        if recipe is None:
            bot.reply_to(message, 'Мне не удалось найти ни одного рецепта из указаных вами ингридиентов((('
                                  '\nСпросите меня рецепт из чего-нибудь другого.')
            # bot.register_next_step_handler(msg, get_text)
        else:
            out_msg = format_recipe(recipe)
            bot.reply_to(message, out_msg, parse_mode='HTML')
            bot.send_message(chat_id,
                             text='Я могу еще чем-то помочь?\nЕсли нет, то попрощайся со мной или напиши /exit')
            # bot.register_next_step_handler(msg, get_text)


def format_recipe(recipe):
    out = f"<b>Название блюда: {recipe['name']}</b>\n\n<b>Все необходимые ингридиенты:</b>\n\n"
    for ingredient in recipe['ingredients']:
        out += f"{ingredient}\n"
    out += f"\n\n<b>Шаги приготовления:</b>\n\n"
    for i, step in enumerate(recipe['steps']):
        out += f"Шаг {i+1}:\n{step}\n\n"
    return out


bot.polling(none_stop=True, interval=0)
