#!/usr/bin/env python3
# shebang line for linux / mac


L1 = [45, 46, 77, 41, 54]
L2 = ['a', 'b', 'c', 'd', 'e']

# for l1, l2 in zip(L1, L2):

#     print('l1 = ' + str(l1))
#     print('l2 = ' + str(l2))


# iterating a list and want to know the index

# for idx, l1 in enumerate(L1):
# print('l1 (idx=' + str(idx) + ') = ' + str(l1))


for idx, (l1, l2) in enumerate(zip(L1, L2)):

    print('idx = ' + str(idx))
    print('l1 = ' + str(l1))
    print('l2 = ' + str(l2))
    print('')
