from handlers import telegramAgent

from settings import TOKEN_SPB

agent = telegramAgent(TOKEN_SPB)
agent.users = {u'Ayrat': 68007066, u'Denis': 95909247}
agent.sendMessage(u'Ayrat', 'Hi')