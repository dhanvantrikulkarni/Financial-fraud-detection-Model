import sqlite3
conn = sqlite3.connect('fraud_detection.db')
cursor = conn.cursor()
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
print('Tables:', cursor.fetchall())
cursor.execute('PRAGMA table_info(transactions)')
print('Columns:', cursor.fetchall())
conn.close()
