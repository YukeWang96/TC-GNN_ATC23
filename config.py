BLK_H = 16 
BLK_W = 8
WARP_SIZE = 32

def func(x):
    if x > 0:
        return x
    else:
        return 1