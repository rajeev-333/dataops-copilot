import sqlite3
import pandas as pd
import os
import random
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data.db')

def create_and_populate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id TEXT NOT NULL,
            location TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            voltage REAL,
            current REAL,
            frequency REAL,
            temperature REAL,
            status TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensors (
            sensor_id TEXT PRIMARY KEY,
            location TEXT,
            type TEXT,
            installed_date TEXT,
            is_active INTEGER
        )
    ''')

    sensors = [
        ('S001', 'Substation_A', 'Voltage_Monitor', '2022-01-15', 1),
        ('S002', 'Substation_B', 'Current_Monitor', '2022-03-20', 1),
        ('S003', 'Substation_C', 'Frequency_Monitor', '2021-11-10', 1),
        ('S004', 'Substation_D', 'Voltage_Monitor', '2023-05-05', 1),
        ('S005', 'Substation_E', 'Temperature_Monitor', '2022-08-30', 0),
    ]
    cursor.executemany('INSERT OR IGNORE INTO sensors VALUES (?,?,?,?,?)', sensors)

    random.seed(42)
    readings = []
    base_time = datetime.now() - timedelta(days=30)

    for i in range(500):
        ts = base_time + timedelta(hours=i)
        sensor = random.choice(sensors)
        voltage = round(random.gauss(230, 5), 2)
        if random.random() < 0.05:
            voltage = round(random.uniform(260, 290), 2)  # spike
        current = round(random.gauss(15, 2), 2)
        frequency = round(random.gauss(50, 0.3), 2)
        temperature = round(random.gauss(45, 5), 2)
        status = 'ANOMALY' if voltage > 255 or temperature > 60 else 'NORMAL'
        readings.append((sensor[0], sensor[1], ts.strftime('%Y-%m-%d %H:%M:%S'),
                         voltage, current, frequency, temperature, status))

    cursor.executemany('''
        INSERT INTO sensor_readings 
        (sensor_id, location, timestamp, voltage, current, frequency, temperature, status)
        VALUES (?,?,?,?,?,?,?,?)
    ''', readings)

    conn.commit()
    conn.close()
    print(f"✅ Database created at: {os.path.abspath(DB_PATH)}")
    print(f"✅ Inserted {len(readings)} sensor readings across {len(sensors)} sensors")

if __name__ == '__main__':
    create_and_populate()
