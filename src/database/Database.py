import sqlite3

class Database:

    def __init__(self):
        self.conn = sqlite3.connect('../src/database/dmbal_db.sqlite')
        self.cur = self.conn.cursor()
        self.initDatasetDB()
        self.initCifar10DB()

    def initDatasetDB(self):
        # Create table
        self.cur.execute('DROP TABLE IF EXISTS datasets')
        self.cur.execute(
            'CREATE TABLE IF NOT EXISTS datasets (dataset_id INTEGER, dataset_name VARCHAR, accuracy REAL, PRIMARY KEY (dataset_id))')
        self.conn.commit()

        #Create columns
        self.setColumnsDatasetDB()

    def initCifar10DB(self):
        # Create table
        self.cur.execute('DROP TABLE IF EXISTS cifar10_classes')
        self.cur.execute(
            'CREATE TABLE IF NOT EXISTS cifar10_classes (class_id INTEGER, class_name VARCHAR, accuracy REAL, PRIMARY KEY (class_id))')
        self.setColumnsCifar10DB()
        self.conn.commit()

    def setColumnsDatasetDB(self):
        self.cur.execute('INSERT INTO datasets (dataset_name) VALUES ("cifar10"), ("mnist")')
        self.conn.commit()

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def setColumnsCifar10DB(self):
        self.cur.execute('INSERT INTO cifar10_classes (class_name) '
                         'VALUES ("plane"), ("car"), ("bird"), ("cat"), ("deer"), '
                         '("dog"), ("frog"), ("horse"), ("ship"), ("truck")')
        self.conn.commit()

    def getCur(self):
        return self.cur

    def closeConnection(self):
        self.conn.close()