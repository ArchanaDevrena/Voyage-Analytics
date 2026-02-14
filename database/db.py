import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    return psycopg2.connect(DATABASE_URL)

##-----new------------

def insert_user_hotel(user_code, name, place, days, price, total, date):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO user_hotels
        (user_code, hotel_name, place, days, price, total, date)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
    """, (user_code, name, place, days, price, total, date))

    conn.commit()
    cur.close()
    conn.close()

