import pandas as pd
from src.config import Config
from src.utils.conn_manager import create_connection

with create_connection() as conn:
    df = pd.read_sql('select * from something',conn)
print(df)