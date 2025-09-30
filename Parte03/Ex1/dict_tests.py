#!/usr/bin/env python3
# shebang line for linux / mac

# initialization of a dict
d_joao = {'name': 'joao', 'age': 21, 'hair': 'black'}

# Append to a dictionary
d_joao['town'] = 'fafe'

# read from  a dictionary
town = d_joao['town']
print('Joao is from ' + town)


# Lists
students = ['joao', 'mateus', 'rita']

print('students = ' + str(students))

# Add one new key in the dict that contains a list
d_joao['friends'] = ['pedro', 'rita', 'gonçalo']


print('d_joao = ' + str(d_joao))


# For rita
d_rita = {'name': 'rita', 'age': 21, 'hair': 'brown'}

# Append to a dictionary
d_rita['town'] = 'viana'

# read from  a dictionary
town = d_rita['town']
print('rita is from ' + town)


# List
students = ['rita', 'mateus', 'rita']

print('students = ' + str(students))

# Add one new key in the dict that contains a list
d_rita['friends'] = ['pedro', 'rita', 'diniz']


print('d_rita = ' + str(d_rita))


# creata a list of dictionaires

d_students = [d_joao, d_rita]
print('d_students' + str(d_students))
