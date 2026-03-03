import os

from gamelens.db import DatabaseConnection


async def init_db():
    print("Initializing DB conncetion pool...")
    conn_str = os.environ.get("PGSQL_CONN")
    if not conn_str:
        raise ValueError("PGSQL_CONN environment variable is not set")

    await DatabaseConnection.initialize(conn_str)
