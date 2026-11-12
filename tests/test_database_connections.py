import sqlite3

import pytest

from greenrouting.database import DatabaseManager


def test_get_connection_closes_sqlite_connection_when_context_exits(tmp_path):
    database = DatabaseManager(tmp_path / "connections.db")

    with database.get_connection() as connection:
        connection.execute("SELECT 1")

    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connection.execute("SELECT 1")
