import sqlite3
import pandas as pd
conn = sqlite3.connect('fraud_detection.db')
df = pd.read_sql('SELECT * FROM transactions LIMIT 5', conn)
print("Data sample:")
print(df)
print("\nColumn types:")
print(df.dtypes)
print("\nColumns:", df.columns.tolist())
conn.close()
