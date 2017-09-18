# coding: utf-8

import natasha.extractors

from yargy import (
    rule, fact,
    or_, and_
)
from yargy.predicates import (
    eq, lte, gte, gram,
    length_eq,
    in_, in_caseless, dictionary,
    normalized, caseless,
    is_title
)
from yargy.pipelines import (
    MorphPipeline
)


Address = fact(
    'Address',
    ['index', 'strana', 'fed_okrug', 'respublika',
     'krai', 'oblast', 'auto_okrug', 'raion',
     'gorod', 'selo', 'derevnya', 'poselok',
     'street', 'prospekt', 'proezd', 'pereulok',
     'ploshad', 'shosse',
     'dom', 'korpus', 'stroenie', 'ofis', 'kvartira']
)


DASH = eq('-')
DOT = eq('.')

ADJF = gram('ADJF')
NOUN = gram('NOUN')
INT = gram('INT')
TITLE = is_title()

ANUM = rule(
    INT,
    DASH.optional(),
    in_caseless({
        'я', 'й', 'е',
        'ое', 'ая', 'ий', 'ой'
    })
)


#########
#
#  STRANA
#
##########


# TODO
STRANA_VALUE = dictionary({
    'россия',
    'украина'
}).interpretation(
    Address.strana
)

ABBR_STRANA_VALUE = in_caseless({
    'рф'
})

STRANA = or_(
    STRANA_VALUE,
    rule(ABBR_STRANA_VALUE)
)


#############
#
#  FED OKRUGA
#
############


FED_OKRUG_NAME = or_(
    rule(
        dictionary({
            'дальневосточный',
            'приволжский',
            'сибирский',
            'уральский',
            'центральный',
            'южный',
        })
    ),
    rule(
        caseless('северо'),
        DASH.optional(),
        dictionary({
            'западный',
            'кавказский'
        })
    )
).interpretation(
    Address.fed_okrug
)

FED_OKRUG_WORDS = or_(
    rule(
        normalized('федеральный'),
        normalized('округ')
    ),
    rule(caseless('фо'))
)

FED_OKRUG = rule(
    FED_OKRUG_WORDS,
    FED_OKRUG_NAME
)


#########
#
#   RESPUBLIKA
#
############


RESPUBLIKA_WORDS = or_(
    rule(caseless('респ'), DOT.optional()),
    rule(normalized('республика'))
)

RESPUBLIKA_ADJF = or_(
    rule(
        dictionary({
            'удмуртский',
            'чеченский',
            'чувашский',
        })
    ),
    rule(
        caseless('карачаево'),
        DASH.optional(),
        normalized('черкесский')
    ),
    rule(
        caseless('кабардино'),
        DASH.optional(),
        normalized('балкарский')
    )
).interpretation(
    Address.respublika
)

RESPUBLIKA_NAME = or_(
    rule(
        dictionary({
            'адыгея',
            'алтай',
            'башкортостан',
            'бурятия',
            'дагестан',
            'ингушетия',
            'калмыкия',
            'карелия',
            'коми',
            'крым',
            'мордовия',
            'татарстан',
            'тыва',
            'хакасия',
            'саха',
            'якутия',
        })
    ),
    rule(caseless('марий'), caseless('эл')),
    rule(
        normalized('северный'), normalized('осетия'),
        rule('-', normalized('алания')).optional()
    )
).interpretation(
    Address.respublika
)

RESPUBLIKA_ABBR = in_caseless({
    'кбр',
    'кчр',
    'рт',  # Татарстан
}).interpretation(
    Address.respublika
)

RESPUBLIKA = or_(
    rule(RESPUBLIKA_ADJF, RESPUBLIKA_WORDS),
    rule(RESPUBLIKA_WORDS, RESPUBLIKA_NAME),
    rule(RESPUBLIKA_ABBR)
)


##########
#
#   KRAI
#
########


KRAI_WORDS = normalized('край')

KRAI_NAME = dictionary({
    'алтайский',
    'забайкальский',
    'камчатский',
    'краснодарский',
    'красноярский',
    'пермский',
    'приморский',
    'ставропольский',
    'хабаровский',
}).interpretation(
    Address.krai
)

KRAI = rule(
    KRAI_NAME, KRAI_WORDS
)


############
#
#    OBLAST
#
############


OBLAST_WORDS = or_(
    rule(normalized('область')),
    rule(
        caseless('обл'),
        DOT.optional()
    )
)

OBLAST_NAME = dictionary({
    'амурский',
    'архангельский',
    'астраханский',
    'белгородский',
    'брянский',
    'владимирский',
    'волгоградский',
    'вологодский',
    'воронежский',
    'горьковский',
    'ивановский',
    'ивановский',
    'иркутский',
    'калининградский',
    'калужский',
    'камчатский',
    'кемеровский',
    'кировский',
    'костромской',
    'курганский',
    'курский',
    'ленинградский',
    'липецкий',
    'магаданский',
    'московский',
    'мурманский',
    'нижегородский',
    'новгородский',
    'новосибирский',
    'омский',
    'оренбургский',
    'орловский',
    'пензенский',
    'пермский',
    'псковский',
    'ростовский',
    'рязанский',
    'самарский',
    'саратовский',
    'сахалинский',
    'свердловский',
    'смоленский',
    'тамбовский',
    'тверский',
    'томский',
    'тульский',
    'тюменский',
    'ульяновский',
    'челябинский',
    'читинский',
    'ярославский',
}).interpretation(
    Address.oblast
)

OBLAST = rule(
    OBLAST_NAME, OBLAST_WORDS
)


##########
#
#    AUTO OKRUG
#
#############


AUTO_OKRUG_NAME = or_(
    rule(
        dictionary({
            'чукотский',
            'эвенкийский',
            'корякский',
            'ненецкий',
            'таймырский',
            'агинский',
            'бурятский',
        })
    ),
    rule(caseless('коми'), '-', normalized('пермяцкий')),
    rule(caseless('долгано'), '-', normalized('ненецкий')),
    rule(caseless('ямало'), '-', normalized('ненецкий')),
).interpretation(
    Address.auto_okrug
)

AUTO_OKRUG_WORDS = or_(
    rule(
        normalized('автономный'),
        normalized('округ')
    ),
    rule(caseless('ао'))
)

HANTI = rule(
    caseless('ханты'), '-', normalized('мансийский')
).interpretation(
    Address.auto_okrug
)

BURAT = rule(
    caseless('усть'), '-', normalized('ордынский'),
    normalized('бурятский')
).interpretation(
    Address.auto_okrug
)

AUTO_OKRUG = or_(
    rule(AUTO_OKRUG_NAME, AUTO_OKRUG_WORDS),
    rule(
        HANTI,
        AUTO_OKRUG_WORDS,
        '-', normalized('югра')
    ),
    rule(
        BURAT,
        AUTO_OKRUG_WORDS
    )
)


##########
#
#  RAION
#
###########


RAION_WORDS = or_(
    rule(caseless('р'), '-', in_caseless({'он', 'н'})),
    rule(normalized('район'))
)

RAION_SIMPLE_NAME = and_(
    ADJF,
    TITLE
)

RAION_MODIFIERS = rule(
    in_caseless({
        'усть',
        'северо',
        'александрово',
        'гаврилово',
    }),
    DASH.optional(),
    TITLE
)

RAION_COMPLEX_NAME = rule(
    RAION_MODIFIERS,
    RAION_SIMPLE_NAME
)

RAION_NAME = or_(
    rule(RAION_SIMPLE_NAME),
    RAION_COMPLEX_NAME
).interpretation(
    Address.raion
)

RAION = rule(RAION_NAME, RAION_WORDS)


###########
#
#   GOROD
#
###########


# Top 200 Russia cities, cover 75% of population


class ComplexGorodPipeline(MorphPipeline):
    grammemes = {'ComplexGorod'}
    keys = [
        'санкт - петербург',
        'нижний новгород',
        'н . новгород',
        'ростов - на - дону',
        'набережные челны',
        'улан - удэ',
        'нижний тагил',
        'комсомольск - на - амуре',
        'йошкар - ола',
        'старый оскол',
        'великий новгород',
        'южно - сахалинск',
        'петропавловск - камчатский',
        'каменск - уральский',
        'орехово - зуево',
        'сергиев посад',
        'новый уренга',
        'ленинск - кузнецкий',
        'великие лук',
        'каменск - шахтинский',
        'усть - илимск',
        'усолье - сибирский',
        'кирово - чепецк',
    ]


SIMPLE = dictionary({
    'москва',
    'новосибирск',
    'екатеринбург',
    'казань',
    'самар',
    'омск',
    'челябинск',
    'уфа',
    'волгоград',
    'пермь',
    'красноярск',
    'воронеж',
    'саратов',
    'краснодар',
    'тольятти',
    'барнаул',
    'ижевск',
    'ульяновск',
    'владивосток',
    'ярославль',
    'иркутск',
    'тюмень',
    'махачкала',
    'хабаровск',
    'оренбург',
    'новокузнецк',
    'кемерово',
    'рязань',
    'томск',
    'астрахань',
    'пенза',
    'липецк',
    'тула',
    'киров',
    'чебоксары',
    'калининград',
    'брянск',
    'курск',
    'иваново',
    'магнитогорск',
    'тверь',
    'ставрополь',
    'симферополь',
    'белгород',
    'архангельск',
    # 'владимир',
    'севастополь',
    'сочи',
    'курган',
    'смоленск',
    'калуга',
    'чита',
    'орёл',
    # 'волжский',
    'череповец',
    'владикавказ',
    'мурманск',
    'сургут',
    'вологда',
    'саранск',
    'тамбов',
    'стерлитамак',
    'грозный',
    'якутск',
    'кострома',
    'петрозаводск',
    'таганрог',
    'нижневартовск',
    'братск',
    'новороссийск',
    'дзержинск',
    'шахта',
    'нальчик',
    'орск',
    'сыктывкар',
    'нижнекамск',
    'ангарск',
    'балашиха',
    'благовещенск',
    'прокопьевск',
    'химки',
    'псков',
    'бийск',
    'энгельс',
    'рыбинск',
    'балаково',
    'северодвинск',
    'армавир',
    'подольск',
    # 'королёв',
    'сызрань',
    'норильск',
    'златоуст',
    'мытищи',
    'люберцы',
    'волгодонск',
    'новочеркасск',
    'абакан',
    'находка',
    'уссурийск',
    'березники',
    'салават',
    'электросталь',
    'миасс',
    'первоуральск',
    'рубцовск',
    'альметьевск',
    'ковровый',
    'коломна',
    'керчь',
    'майкоп',
    'пятигорск',
    'одинцово',
    'копейск',
    'хасавюрт',
    'новомосковск',
    'кисловодск',
    'серпухов',
    'новочебоксарск',
    'нефтеюганск',
    'димитровград',
    'нефтекамск',
    'черкесск',
    'дербент',
    'камышин',
    'невинномысск',
    'красногорск',
    'мур',
    'батайск',
    'новошахтинск',
    'ноябрьск',
    'кызыл',
    # 'октябрьский',
    'ачинск',
    'северск',
    'новокуйбышевск',
    'елец',
    'евпатория',
    'арзамас',
    'обнинск',
    'каспийск',
    'элиста',
    'пушкино',
    # 'жуковский',
    'междуреченск',
    'сарапул',
    'ессентуки',
    'воткинск',
    'ногинск',
    'тобольск',
    'ухта',
    'серов',
    'бердск',
    'мичуринск',
    'киселёвск',
    'новотроицк',
    'зеленодольск',
    'соликамск',
    'раменский',
    'домодедово',
    'магадан',
    'глазов',
    'железногорск',
    'канск',
    'назрань',
    'гатчина',
    'саров',
    'новоуральск',
    'воскресенск',
    'долгопрудный',
    'бугульма',
    'кузнецк',
    'губкин',
    'кинешма',
    'ейск',
    'реутов',
    'железногорск',
    'чайковский',
    'азов',
    'бузулук',
    'озёрск',
    'балашов',
    'юрга',
    'кропоткин',
    'клин'
})

COMPLEX = gram('ComplexGorod')

GOROD_ABBR = in_caseless({
    'спб',
    'мск'
})

GOROD_NAME = or_(
    SIMPLE,
    COMPLEX,
    GOROD_ABBR
).interpretation(
    Address.gorod
)

SIMPLE = and_(
    TITLE,
    or_(
        NOUN,
        ADJF  # Железнодорожный, Юбилейный
    )
)

COMPLEX = or_(
    rule(
        SIMPLE,
        DASH.optional(),
        SIMPLE
    ),
    rule(
        TITLE,
        DASH.optional(),
        caseless('на'),
        DASH.optional(),
        TITLE
    )
)

NAME = or_(
    rule(SIMPLE),
    COMPLEX
)

MAYBE_GOROD_NAME = or_(
    NAME,
    rule(NAME, '-', INT)
).interpretation(
    Address.gorod
)

GOROD_WORDS = or_(
    rule(normalized('город')),
    rule(
        caseless('г'),
        DOT.optional()
    )
)

GOROD = or_(
    rule(GOROD_WORDS, MAYBE_GOROD_NAME),
    rule(
        GOROD_WORDS.optional(),
        GOROD_NAME
    )
)


##########
#
#  SELO NAME
#
##########


ADJS = gram('ADJS')
SIMPLE = and_(
    or_(
        NOUN,  # Александровка, Заречье, Горки
        ADJS,  # Кузнецово
        ADJF,  # Никольское, Новая, Марьино
    ),
    TITLE
)

COMPLEX = rule(
    SIMPLE,
    DASH.optional(),
    SIMPLE
)

NAME = or_(
    rule(SIMPLE),
    COMPLEX
)

SELO_NAME = or_(
    NAME,
    rule(NAME, '-', INT),
    rule(NAME, ANUM)
)


###########
#
#   SELO
#
#############


SELO_WORDS = or_(
    rule(
        caseless('с'),
        DOT.optional()
    ),
    rule(normalized('село'))
)

SELO = rule(
    SELO_WORDS,
    SELO_NAME.interpretation(
        Address.selo
    )
)


###########
#
#   DEREVNYA
#
#############


DEREVNYA_WORDS = or_(
    rule(
        caseless('д'),
        DOT.optional()
    ),
    rule(normalized('деревня'))
)

DEREVNYA_NAME = SELO_NAME.interpretation(
    Address.derevnya
)

DEREVNYA = rule(
    DEREVNYA_WORDS,
    DEREVNYA_NAME
)


###########
#
#   POSELOK
#
#############


POSELOK_WORDS = or_(
    rule(
        in_caseless({'п', 'пос'}),
        DOT.optional()
    ),
    rule(normalized('посёлок')),
    rule(
        caseless('р'),
        DOT.optional(),
        caseless('п'),
        DOT.optional()
    ),
    rule(
        normalized('рабочий'),
        normalized('посёлок')
    )
)

POSELOK_NAME = SELO_NAME.interpretation(
    Address.poselok
)

POSELOK = rule(
    POSELOK_WORDS,
    POSELOK_NAME
)


##############
#
#   ADDRESS PERSON
#
############


ABBR = and_(
    length_eq(1),
    is_title()
)

PART = and_(
    TITLE,
    or_(
        gram('Name'),
        gram('Surn')
    )
)

MAYBE_FIO = or_(
    rule(TITLE, PART),
    rule(PART, TITLE),
    rule(ABBR, '.', TITLE),
    rule(ABBR, '.', ABBR, '.', TITLE),
    rule(TITLE, ABBR, '.', ABBR, '.')
)

POSITION_WORDS_ = or_(
    rule(
        dictionary({
            'мичман',
            'геолог',
            'подводник',
            'краевед',
            'снайпер',
            'штурман',
            'бригадир',
            'учитель',
            'политрук',
            'военком',
            'ветеран',
            'историк',
            'пулемётчик',
            'авиаконструктор',
            'адмирал',
            'академик',
            'актер',
            'актриса',
            'архитектор',
            'атаман',
            'врач',
            'воевода',
            'генерал',
            'губернатор',
            'хирург',
            'декабрист',
            'разведчик',
            'граф',
            'десантник',
            'конструктор',
            'скульптор',
            'писатель',
            'поэт',
            'капитан',
            'князь',
            'комиссар',
            'композитор',
            'космонавт',
            'купец',
            'лейтенант',
            'лётчик',
            'майор',
            'маршал',
            'матрос',
            'подполковник',
            'полковник',
            'профессор',
            'сержант',
            'старшина',
            'танкист',
            'художник',
            'герой',
            'княгиня',
            'строитель',
            'дружинник',
            'диктор',
            'прапорщик',
            'артиллерист',
            'графиня',
            'большевик',
            'патриарх',
            'сварщик',
            'офицер',
            'рыбак',
            'брат',
        })
    ),
    rule(normalized('генерал'), normalized('армия')),
    rule(normalized('герой'), normalized('россия')),
    rule(
        normalized('герой'),
        normalized('российский'), normalized('федерация')),
    rule(
        normalized('герой'),
        normalized('советский'), normalized('союз')
    ),
)

ABBR_POSITION_WORDS = rule(
    in_caseless({
        'адм',
        'ак',
        'акад',
    }),
    DOT.optional()
)

POSITION_WORDS = or_(
    POSITION_WORDS_,
    ABBR_POSITION_WORDS
)

MAYBE_PERSON = or_(
    MAYBE_FIO,
    rule(POSITION_WORDS, MAYBE_FIO),
    rule(POSITION_WORDS, TITLE)
)


###########
#
#   IMENI
#
##########


IMENI_WORDS = or_(
    rule(
        caseless('им'),
        DOT.optional()
    ),
    rule(caseless('имени'))
)

IMENI = or_(
    rule(
        IMENI_WORDS.optional(),
        MAYBE_PERSON
    ),
    rule(
        IMENI_WORDS,
        TITLE
    )
)

##########
#
#   LET
#
##########


LET_WORDS = or_(
    rule(caseless('лет')),
    rule(
        DASH.optional(),
        caseless('летия')
    )
)

LET_NAME = in_caseless({
    'влксм',
    'ссср',
    'алтая',
    'башкирии',
    'бурятии',
    'дагестана',
    'калмыкии',
    'колхоза',
    'комсомола',
    'космонавтики',
    'москвы',
    'октября',
    'пионерии',
    'победы',
    'приморья',
    'района',
    'совхоза',
    'совхозу',
    'татарстана',
    'тувы',
    'удмуртии',
    'улуса',
    'хакасии',
    'целины',
    'чувашии',
    'якутии',
})

LET = rule(
    INT,
    LET_WORDS,
    LET_NAME
)


##########
#
#    ADDRESS DATE
#
#############


MONTH_WORDS = dictionary({
    'январь',
    'февраль',
    'март',
    'апрель',
    'май',
    'июнь',
    'июль',
    'август',
    'сентябрь',
    'октябрь',
    'ноябрь',
    'декабрь',
})

DAY = and_(
    INT,
    gte(1),
    lte(31)
)

YEAR = and_(
    INT,
    gte(1),
    lte(2100)
)

YEAR_WORDS = normalized('год')

DATE = or_(
    rule(DAY, MONTH_WORDS),
    rule(YEAR, YEAR_WORDS)
)


#########
#
#   MODIFIER
#
############


MODIFIER_WORDS_ = rule(
    dictionary({
        'большой',
        'малый',
        'средний',

        'верхний',
        'центральный',
        'нижний',
        'северный',
        'дальний',

        'первый',
        'второй',

        'старый',
        'новый',

        'красный',
        'лесной',
        'тихий',
    }),
    DASH.optional()
)

ABBR_MODIFIER_WORDS = rule(
    in_caseless({
        'б', 'м', 'н'
    }),
    DOT.optional()
)

SHORT_MODIFIER_WORDS = rule(
    in_caseless({
        'больше',
        'мало',
        'средне',

        'верх',
        'верхне',
        'центрально',
        'нижне',
        'северо',
        'дальне',
        'восточно',
        'западно',

        'перво',
        'второ',

        'старо',
        'ново',

        'красно',
        'тихо',
        'горно',
    }),
    DASH.optional()
)

MODIFIER_WORDS = or_(
    MODIFIER_WORDS_,
    ABBR_MODIFIER_WORDS,
    SHORT_MODIFIER_WORDS,
)


##########
#
#   ADDRESS NAME
#
##########


ROD = gram('gent')

SIMPLE = and_(
    or_(
        ADJF,  # Школьная
        and_(NOUN, ROD),  # Ленина, Победы
    ),
    TITLE
)

COMPLEX = or_(
    rule(
        and_(ADJF, TITLE),
        NOUN
    ),
    rule(
        TITLE,
        DASH.optional(),
        TITLE
    ),
)

# TODO
EXCEPTION = dictionary({
    'арбат'
})

MAYBE_NAME = or_(
    rule(SIMPLE),
    COMPLEX,
    rule(EXCEPTION)
)

NAME = or_(
    MAYBE_NAME,
    LET,
    DATE,
    IMENI
)

NAME = rule(
    MODIFIER_WORDS.optional(),
    NAME
)

NAME = or_(
    NAME,
    ANUM,
    rule(NAME, ANUM),
    rule(ANUM, NAME),
    rule(INT, DASH.optional(), NAME),
    rule(NAME, DASH, INT),
)

ADDRESS_NAME = NAME


########
#
#    STREET
#
#########


STREET_WORDS = or_(
    rule(normalized('улица')),
    rule(
        caseless('ул'),
        DOT.optional()
    )
)

STREET_NAME = ADDRESS_NAME.interpretation(
    Address.street
)

STREET = or_(
    rule(STREET_WORDS, STREET_NAME),
    rule(STREET_NAME, STREET_WORDS)
)


##########
#
#    PROSPEKT
#
##########


PROSPEKT_WORDS = or_(
    rule(
        in_caseless({'пр', 'просп'}),
        DOT.optional()
    ),
    rule(
        caseless('пр'),
        '-',
        in_caseless({'кт', 'т'}),
        DOT.optional()
    ),
    rule(normalized('проспект'))
)

PROSPEKT_NAME = ADDRESS_NAME.interpretation(
    Address.prospekt
)

PROSPEKT = or_(
    rule(PROSPEKT_WORDS, PROSPEKT_NAME),
    rule(PROSPEKT_NAME, PROSPEKT_WORDS)
)


############
#
#    PROEZD
#
#############


PROEZD_WORDS = or_(
    rule(caseless('пр'), DOT.optional()),
    rule(
        caseless('пр'),
        '-',
        in_caseless({'зд', 'д'}),
        DOT.optional()
    ),
    rule(normalized('проезд'))
)

PROEZD_NAME = ADDRESS_NAME.interpretation(
    Address.proezd
)

PROEZD = or_(
    rule(PROEZD_WORDS, PROEZD_NAME),
    rule(PROEZD_NAME, PROEZD_WORDS)
)


###########
#
#   PEREULOK
#
##############


PEREULOK_WORDS = or_(
    rule(
        in_caseless({'п', 'пер'}),
        DOT.optional()
    ),
    rule(normalized('переулок'))
)

PEREULOK_NAME = ADDRESS_NAME.interpretation(
    Address.pereulok
)

PEREULOK = or_(
    rule(PEREULOK_WORDS, PEREULOK_NAME),
    rule(PEREULOK_NAME, PEREULOK_WORDS)
)


########
#
#  PLOSHAD
#
##########


PLOSHAD_WORDS = or_(
    rule(
        caseless('пл'),
        DOT.optional()
    ),
    rule(normalized('площадь'))
)

PLOSHAD_NAME = ADDRESS_NAME.interpretation(
    Address.ploshad
)

PLOSHAD = or_(
    rule(PLOSHAD_WORDS, PLOSHAD_NAME),
    rule(PLOSHAD_NAME, PLOSHAD_WORDS)
)


############
#
#   SHOSSE
#
###########


# TODO
# Покровское 17 км.
# Сергеляхское 13 км
# Сергеляхское 14 км.


SHOSSE_WORDS = or_(
    rule(
        caseless('ш'),
        DOT.optional()
    ),
    rule(normalized('шоссе'))
)

SHOSSE_NAME = ADDRESS_NAME.interpretation(
    Address.shosse
)

SHOSSE = or_(
    rule(SHOSSE_WORDS, SHOSSE_NAME),
    rule(SHOSSE_NAME, SHOSSE_WORDS)
)


##############
#
#   ADDRESS VALUE
#
#############


LETTER = in_caseless(set('абвгдежзиклмнопрстуфхшщэюя'))

QUOTE = gram('QUOTE')

LETTER = or_(
    rule(LETTER),
    rule(QUOTE, LETTER, QUOTE)
)

VALUE = rule(
    INT,
    LETTER.optional()
)

SEP = in_({'/', '\\', '-'})

VALUE = or_(
    rule(VALUE),
    rule(VALUE, SEP, VALUE),
    rule(VALUE, SEP, LETTER)
)

ADDRESS_VALUE = rule(
    eq('№').optional(),
    VALUE
)

############
#
#    DOM
#
#############


DOM_WORDS = or_(
    rule(normalized('дом')),
    rule(
        caseless('д'),
        DOT.optional()
    )
)

DOM_VALUE = ADDRESS_VALUE.interpretation(
    Address.dom
)

DOM = rule(
    DOM_WORDS,
    DOM_VALUE
)


###########
#
#  KORPUS
#
##########


KORPUS_WORDS = or_(
    rule(
        in_caseless({'корп', 'кор'}),
        DOT.optional()
    ),
    rule(normalized('корпус'))
)

KORPUS_VALUE = ADDRESS_VALUE.interpretation(
    Address.korpus
)

KORPUS = or_(
    rule(
        KORPUS_WORDS,
        KORPUS_VALUE
    ),
    rule(
        KORPUS_VALUE,
        KORPUS_WORDS
    )
)


###########
#
#  STROENIE
#
##########


STROENIE_WORDS = or_(
    rule(
        caseless('стр'),
        DOT.optional()
    ),
    rule(normalized('строение'))
)

STROENIE_VALUE = ADDRESS_VALUE.interpretation(
    Address.stroenie
)

STROENIE = rule(
    STROENIE_WORDS,
    ADDRESS_VALUE
)


###########
#
#   OFIS
#
#############


OFIS_WORDS = or_(
    rule(
        caseless('оф'),
        DOT.optional()
    ),
    rule(normalized('офис'))
)

OFIS_VALUE = ADDRESS_VALUE.interpretation(
    Address.ofis
)

OFIS = rule(
    OFIS_WORDS,
    OFIS_VALUE
)


###########
#
#   KVARTIRA
#
#############


KVARTIRA_WORDS = or_(
    rule(
        caseless('кв'),
        DOT.optional()
    ),
    rule(normalized('квартира'))
)

KVARTIRA_VALUE = ADDRESS_VALUE.interpretation(
    Address.kvartira
)

KVARTIRA = rule(
    KVARTIRA_WORDS,
    KVARTIRA_VALUE
)


###########
#
#   INDEX
#
#############


INDEX = and_(
    INT,
    gte(100000),
    lte(999999)
).interpretation(
    Address.index
)


#############
#
#   FULL ADDRESS
#
############


OBLAST_LEVEL = or_(
    RESPUBLIKA,
    KRAI,
    OBLAST,
    AUTO_OKRUG
)
GOROD_LEVEL = or_(
    GOROD,
    DEREVNYA,
    SELO,
    POSELOK
)
STREET_LEVEL = or_(
    STREET,
    PROSPEKT,
    PROEZD,
    PEREULOK,
    PLOSHAD,
    SHOSSE
)
OFIS_LEVEL = or_(
    OFIS,
    KVARTIRA
)

SEP = in_({',', ';'})

ADDRESS = rule(
    rule(
        INDEX,
        SEP.optional()
    ).optional(),

    rule(
        STRANA,
        SEP.optional()
    ).optional(),

    rule(
        OBLAST_LEVEL,
        SEP.optional()
    ).optional(),

    rule(
        RAION,
        SEP.optional()
    ).optional(),

    rule(
        GOROD_LEVEL,
        SEP.optional()
    ).optional(),

    # Москва, Зеленоград
    # TODO overwrites
    rule(
        GOROD_LEVEL,
        SEP.optional()
    ).optional(),

    STREET_LEVEL,
    rule(
        SEP.optional(),
        or_(
            DOM,
            DOM_VALUE
        )
    ).optional(),

    rule(
        SEP.optional(),
        KORPUS
    ).optional(),

    rule(
        SEP.optional(),
        STROENIE
    ).optional(),

    rule(
        SEP.optional(),
        OFIS_LEVEL
    ).optional()
).interpretation(
    Address
)


def extractor():
    return natasha.extractors.Extractor(ADDRESS, [ComplexGorodPipeline()])
