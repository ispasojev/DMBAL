from src.database.Database import Database

database = Database()
cur = database.getCur()

# cur.execute('SELECT * FROM datasets')
# data = cur.fetchall()
# print(data)

classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cur.execute('SELECT * FROM cifar10_classes')
data = cur.fetchall()
print(data)

var = 0.5
cur.execute("UPDATE datasets SET accuracy = ? WHERE dataset_name = 'cifar10'", (var,))
cur.execute('SELECT * FROM datasets')
data = cur.fetchall()
print(data)

for i in range(10):
    acc = round((2 / 3), 3)
    cur.execute("UPDATE cifar10_classes SET accuracy = ? WHERE class_name = ?", (acc, classes[i],))
    print("hi")
cur.execute('SELECT * FROM cifar10_classes')
data = cur.fetchall()
print(data)