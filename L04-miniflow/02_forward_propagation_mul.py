"""
No need to change anything here!

If all goes well, this should work after you
modify the Add class in miniflow.py.
"""

from miniflow_02 import *

x, y, z = Input(), Input(), Input()
a, b, c = Input(), Input(), Input()

f = Add(x, y, z)
#g = Mul(a, b, c)

feed_dict_add = {x: 4, y: 5, z: 10}
feed_dict_mul = {a: 4, b: 5, c: 10}

graph = topological_sort(feed_dict_add)
output = forward_pass(f, graph)
# should output 19
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))

graph = topological_sort(feed_dict_mul)
#output = forward_pass(g, graph)
# should output 200
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
