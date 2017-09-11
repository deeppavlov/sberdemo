class Sayer:
    @classmethod
    def say(cls, method_name, ctx):
        return getattr(cls, method_name)(ctx)

    @staticmethod
    def cant_reserve():
        return 'Нельзя резервировать счёт не в рублях'

    @staticmethod
    def new_acc_documents_list(ctx):
        return 'Список документов для {}резидента РФ'.format('' if ctx['resident'] else 'не ')

    @staticmethod
    def new_acc_rates_list(ctx):
        return 'Тарифы для {}резидента РФ'.format('' if ctx['resident'] else 'не ')

    @staticmethod
    def not_supported(ctx):
        return "Такая валюта не поддерживается. Можно открыть в рублях, долларах и евро"

    @staticmethod
    def send_to_bank(ctx):
        return "Для открытия счёта обратись в отеление Сбербанка"

    @staticmethod
    def reserve_new_acc_online(ctx):
        return "`Ссылка на онлайн-резервирование`"

    @staticmethod
    def weird_route(ctx):
        return 'You were not supposed to see this'

    @staticmethod
    def show_vsp(ctx):
        return '`точки на карте с отеделениями`'

    @staticmethod
    def what_now(ctx):
        return 'Мы можем вам ещё как-нибудь помочь?'

    @staticmethod
    def no_intent(ctx):
        return 'Простите, не поняла'
