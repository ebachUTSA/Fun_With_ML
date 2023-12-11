import urllib.parse
from sqlalchemy import create_engine
from src.config import Config
config = Config()

def create_connection(database=config.dac_db):
    """Create a connection string for SQL Server."""
    if config.env_type == "PROD":
        conn_string = f"mssql+pyodbc://{config.uid}:{urllib.parse.quote_plus(config.pid)}@{config.server}/{database}?driver={urllib.parse.quote_plus(config.driver)}&Encrypt=no"
    else:
        conn_string = f"mssql://@{config.server}/{database}?driver={config.driver}&Encrypt=no"
    engine = create_engine(conn_string)
    return engine.begin()
