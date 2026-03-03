from psycopg_pool import AsyncConnectionPool


class DatabaseConnection:
    """
    A Singleton wrapper around a Connection Pool.
    """

    _pool = None

    @classmethod
    async def initialize(cls, conn_string):
        if cls._pool is None:
            print("Initializing Connection Pool...")
            cls._pool = AsyncConnectionPool(
                conn_string,
                min_size=1,
                max_size=25,
                open=True,  # Pre-open connections to avoid boot-time hangs
            )
            await cls._pool.open()

    @classmethod
    def get_connection(cls):
        """
        Returns a context manager that yields a connection from the pool.
        """
        if cls._pool is None:
            raise Exception(
                "Database not initialized. Call Database.initialize() first."
            )

        # Usage: async with Database.get_connection() as conn: ...
        return cls._pool.connection()

    @classmethod
    async def close(cls):
        """
        Clean up the pool on app shutdown.
        """
        if cls._pool:
            await cls._pool.close()
            print("Connection Pool closed.")
