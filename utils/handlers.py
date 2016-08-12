#!/usr/bin/python
# -*- coding: utf-8 -*-

import pymysql
import sys
from telegram.ext import Updater


class mysqlHandler:
    def __init__(self, host, port, username, passwd, db):
        self.host = host
        self.port = port
        self.username = username
        self.passcode = passwd
        self.db = db
        self.conn = None
        self.cur = None

    def __repr__(self):
        return "<MySQL Handler to ('%s-%s')>" % (self.host, self.db)

    def check_response(funk):
        def func_wrapper(*args, **kwargs):
            try:
                return funk(*args, **kwargs)
            except pymysql.OperationalError as e:
                error, msg = e
                if error == 2003:
                    print ('Error: No Route to Host')
                    print ('Connection terminated')
                    sys.exit()
                elif error == 1045:
                    print ('Error: Access denied for user')
                    print ('Connection terminated')
                    sys.exit()
                elif error == 2006 or error == 2013:
                    print (e)
                    print ("Reconnecting...")
                    return funk(*args, **kwargs)
        return func_wrapper

    @check_response
    def _execute(self, query, close_connection):
        self.conn = pymysql.connect(host=self.host, port=self.port, user=self.username, passwd=self.passcode, db=self.db)
        self.cur = self.conn.cursor()
        self.cur.execute(query)
        r = [i for i in self.cur.fetchall()]
        if close_connection:
            self.cur.close()
            self.conn.close()
        return r

    def get_results(self, query, close_connection=True):
        return self._execute(query, close_connection)



class telegramAgent:

    def __init__(self, token):
        self.token = token
        temp = Updater(token=token)
        self.updater = temp
        self.users = {}

    def botDetails(self):
        print (self.updater.bot.getMe())

    def getUpdate(self):
        return self.updater.bot.getUpdates(limit = 50)

    def sendMessage(self, name, message):
        id = self.users[name]
        self.updater.bot.sendMessage(chat_id=id, text=message)

    def checkAllUsers(self):
        updates = self.updater.bot.getUpdate()
        print (set([u.message.chat_id for u in updates]))


