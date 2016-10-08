"""
Utility function for saving results into a PostgreSQL database.
"""
import numpy as np
import psycopg2 as pg

from contextlib import contextmanager

__author__ = 'alshedivat'


@contextmanager
def database_cursor(user='nig',
                    password='agreement332',
                    database='nig',
                    host='localhost'):
    conn = psycopg2.connect("user='{user:s}' password='{password:s}'"
                            "dbname='{database:s}' host='{host:s}'"
                            .format(user=user, password=password,
                                    database=database, host=host))
    curs = conn.cursor()
    yield curs
    curs.close()
    conn.close()


def create_table(params, results, table,
                 user='nig',
                 password='agreement332',
                 database='nig',
                 host='localhost'):
    """Create a table in the database with corresponding columns.

    Arguments:
    ----------
        param_names : dict
        result_names : dict
        user : str (default: 'nig')
        password : str (default: 'agreement332')
        database : str (default: 'nig')
        host : str (default: 'localhost')
    """
    SQL = """
    CREATE TABLE {table:s} (
        id serial PRIMARY KEY,
        time timestamp NOT NULL""".format(table=table)
    for pname, ptype in params.iteritems():
        SQL += ",\n%s %s" % (pname, ptype)
    for rname, rtype in results.iteritems():
        SQL += ",\n%s %s" % (rname, rtype)
    SQL += ");"
    with database_cursor(user, password, database, host) as curs:
        curs.execute(SQL)


def save(params, results, table,
         user='nig',
         password='agreement332',
         database='nig',
         host='localhost'):
    """Save the given parameters and results into the database.

    Arguments:
    ----------
        params : dict
        results : dict
        user : str (default: 'nig')
        password : str (default: 'agreement332')
        database : str (default: 'nig')
        host : str (default: 'localhost')
    """
    SQL = "INSERT INTO {table:s} (".format(table=table)
    for pname in params.iterkeys():
        SQL += "%s, " % pname
    for rname in results.iterkeys():
        SQL += "%s, " % rname
    SQL += "\b\b) VALUES ("
    for pval in params.itervalues():
        SQL += "%s, " % pval
    for rval in results.itervalues():
        SQL += "%s, " % rval
    SQL += "\b\b);"
    with database_cursor(user, password, database, host) as curs:
        curs.execute(SQL)
