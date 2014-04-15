import sqlite3

class DatabaseManager(object):
    '''Singleton Container for the various databases we'll be using'''
    _relay_db_info = {'relay_main': 'relay.db'}

    def __init__(self, **kwargs):
        db_info = kwargs.get('db_info', DatabaseManager._relay_db_info)
        self.connections = {}
        for name, f in db_info.iteritems():
            self.connections[name] = sqlite3.connect(f, check_same_thread=False)

    def prepare_cursor(self, db_name, query, options=None):
        '''Returns executed cursor'''
        conn = self.connections[db_name]
        cur = conn.cursor()
        if sqlite3.complete_statement(query):
            try:
                query = query.strip()
                if options==None:
                    cur.execute(query)
                else:
                    cur.execute(query, options)
            except sqlite3.Error as e:
                print 'Bad news: ', str(e)
        else:
            raise ValueError('""%s"" is not a valid SQL Statement' % query)
        return cur

    def iter_query(self, db_name, query, options=None, as_dict=False, commit=False):
        '''Returns an iterator or dict over the query results.
        *as_dict: flag to format rows as dictionaries,
        *commit: call connection.commit() after query execution'''
        cur = self.prepare_cursor(db_name, query, options)
        if commit:
           self.commit(db_name)
        if as_dict:
            for line in cur:
                yield dict((cur.description[i][0], value) for i, value in
                    enumerate(line))
        else:
            for line in cur:
                yield line

    def query(self, db_name, query, options=None, as_dict=False, commit=False):
        '''Returns entire result set.
        *as_dict: flag to format rows as dictionaries,
        *commit: call connection.commit() after query execution'''
        cur = self.prepare_cursor(db_name, query, options)
        results = cur.fetchall()
        if commit:
           self.commit(db_name)
        if as_dict:
            return [dict((cur.description[i][0], value) for i, value in
            enumerate(row)) for row in results]
        else:
            return results

    def close_all(self):
        '''Close all open Database Connections'''
        for k in self.connections:
            self.connections[k].close()

    def commit(self, db_name):
        '''Call .commit() on the connection to db_name'''
        return self.connections[db_name].commit()
