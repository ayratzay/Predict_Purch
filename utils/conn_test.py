import pymysql
from utils.settings import HOST, PORT, USERNAME, PASSCODE_VIR, PASSCODE_DEBUG, CAT_DB
from utils.handlers import mysqlHandler

try:
    db = pymysql.connect(host=HOST, port=PORT, user=USERNAME, passwd=PASSCODE_DEBUG, db=CAT_DB)
    cursor = db.cursor()
    cursor.execute("SELECT VERSION()")
    results = cursor.fetchone()
    if results:
        print ('CONNECTED')
    else:
        print ('RESULTS ARRAY IS EMPTY')
except pymysql.Error:
    print ('ERROR IN CONNECTION')


piratecats = mysqlHandler(HOST, PORT, USERNAME, PASSCODE_DEBUG, CAT_DB)
aa = piratecats.get_results("SELECT VERSION()")
