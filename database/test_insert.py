from db import get_connection

conn = get_connection()
cur = conn.cursor()

cur.execute("""
INSERT INTO users (user_code, name, age, gender, company, password)
VALUES ('U1002','Test User',23,'female','Demo','password123')
""")

conn.commit()
cur.close()
conn.close()

print("Inserted once")
