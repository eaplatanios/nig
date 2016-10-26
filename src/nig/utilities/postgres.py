# Copyright 2016, The NIG Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Utility function for saving results into a PostgreSQL database."""
from __future__ import absolute_import, division, print_function

import time
import psycopg2 as pg

from contextlib import contextmanager

__author__ = 'alshedivat'


@contextmanager
def database_cursor(user='nig',
                    password='agreement332',
                    database='nig',
                    host='localhost'):
    db = pg.connect("user='{user:s}' password='{password:s}'"
                    "dbname='{database:s}' host='{host:s}'"
                    .format(user=user, password=password,
                            database=database, host=host))
    curs = db.cursor()
    yield curs
    curs.close()
    db.commit()
    db.close()


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
    SQL_DROP = "DROP TABLE IF EXISTS {table:s};" \
               .format(table=table)
    SQL_CREATE = "CREATE TABLE {table:s} (" \
                 "id serial PRIMARY KEY,"   \
                 "time timestamp NOT NULL"  \
                 .format(table=table)
    for pname, ptype in params.iteritems():
        SQL_CREATE += ",\n%s %s" % (pname, ptype)
    for rname, rtype in results.iteritems():
        SQL_CREATE += ",\n%s %s" % (rname, rtype)
    SQL_CREATE += ");"
    with database_cursor(user, password, database, host) as curs:
        curs.execute(SQL_DROP)
        curs.execute(SQL_CREATE)


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
        table : str
        user : str (default: 'nig')
        password : str (default: 'agreement332')
        database : str (default: 'nig')
        host : str (default: 'localhost')
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    SQL = "INSERT INTO {table:s} (".format(table=table)
    SQL += ', '.join(['time'] + params.keys() + results.keys())
    SQL += ")\nVALUES ("
    SQL += ', '.join(['%s'] * (1 + len(params) + len(results)))
    SQL += ")"
    with database_cursor(user, password, database, host) as curs:
        curs.execute(SQL, [timestamp] + params.values() + results.values())

