from .nlu import ClassifierSlot, DictionarySlot

NEW_ACC_RESERVE_ONLINE = 'NEW_ACC_RESERVE_ONLINE'
NEW_ACC_CURRENCY = 'NEW_ACC_CURRENCY'
CLIENT_RF_RESIDENT = 'CLIENT_RF_RESIDENT'
NEW_ACC_SHOW_DOCS = 'NEW_ACC_SHOW_DOCS'
NEW_ACC_SHOW_RATES = 'NEW_ACC_SHOW_RATES'
NEW_ACC_OWNERSHIP_FORM = 'NEW_ACC_OWNERSHIP_FORM'
NEW_ACC_REGION = 'NEW_ACC_REGION'

SEARCH_VSP = 'search_vsp'
METHOD_LOCATION = 'method_location'

slot_objects = {
    NEW_ACC_RESERVE_ONLINE: ClassifierSlot(NEW_ACC_RESERVE_ONLINE, 'Хотите зарезервировать онлайн?', dict(), dict()),
    NEW_ACC_CURRENCY: DictionarySlot(NEW_ACC_CURRENCY, 'В какой валюте?', dict(), dict()),
    CLIENT_RF_RESIDENT: ClassifierSlot(CLIENT_RF_RESIDENT, 'Вы резидент РФ?', dict(), dict()),
    NEW_ACC_SHOW_DOCS: ClassifierSlot(NEW_ACC_SHOW_DOCS, 'Хотите на документы посмотреть??', dict(), dict()),
    NEW_ACC_SHOW_RATES: ClassifierSlot(NEW_ACC_SHOW_RATES, 'Хотите на тарифы посмотреть??', dict(), dict()),
    NEW_ACC_OWNERSHIP_FORM: DictionarySlot(NEW_ACC_OWNERSHIP_FORM, 'Какая форма собственности?', dict(), dict()),
    NEW_ACC_REGION: DictionarySlot(NEW_ACC_REGION, 'В каком регионе живёте?', dict(), dict()),

    SEARCH_VSP: ClassifierSlot(SEARCH_VSP, 'Найти тебе отделение?', dict(), dict()),
    METHOD_LOCATION: DictionarySlot(METHOD_LOCATION, 'Где искать? Можешь указать метро, ближайший адрес или координаты',
                                    dict(), dict())
}

SUPPORTED_CURRENCIES = ['RUB', 'EUR', 'USD']
slot_objects[NEW_ACC_CURRENCY].filters['supported_currency'] = lambda x, _: x in SUPPORTED_CURRENCIES
slot_objects[NEW_ACC_CURRENCY].filters['not_supported_currency'] = lambda x, _: x not in SUPPORTED_CURRENCIES
