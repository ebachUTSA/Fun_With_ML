from sqlalchemy import create_engine
from src.config import Config
config = Config()

def create_connection():
    """Create a connection string for SQL Server."""
    conn_string = f"mysql+pymysql://{config.uid}:{config.pid}@{config.server}/{config.database}"
    engine = create_engine(conn_string)
    return engine.begin()
