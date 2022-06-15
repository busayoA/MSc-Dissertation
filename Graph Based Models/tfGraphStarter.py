import tensorflow as tf

tf.compat.v1.disable_eager_execution()
merge = "/Users/olubusayoakeredolu/Library/Mobile Documents/com~apple~CloudDocs/GitHub/Dissertation/Data/Sorting/Merge Sort/1.py"

a = 2
b = 3
c = tf.add(a, b, name='Add')

sess = tf.compat.v1.Session()
print(sess.run(c))
sess.close()

with open(merge) as f:
    m = f.read()

x = tf.function(m)

print(x)
