import os
for i in range(10000):
    if len(os.listdir("./{}/".format(i)))<200:
        print( i)
