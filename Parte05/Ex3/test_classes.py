#!/usr/bin/env python3

class Animal:

    def __init__(self, name, age):  # Constructor. It runs every time I create an instance of an animal

        # storing the transient varieable name into a class property (not transient) self.name
        self.name = name
        self.age = age
        print('Running the constructor for the Animal class')

    def __str__(self):  # returns a text that will be used when the instance of the class is printted
        return "Class name=" + self.name + ' age=' + str(self.age)

    def getYearsMissingTo100(self):
        return 100 - self.age


def main():

    ##########################################
    # Creatign animals with separate variables
    ##########################################
    # Definition
    name1 = 'bobi'
    age1 = 2

    name2 = 'lassie'
    age2 = 79

    # printing
    # I want to print all the infotmation about an animal
    print('name = ' + name1 + ' age = ' + str(age1))
    print('name = ' + name2 + ' age = ' + str(age2))

    # Get years to 100
    y100 = 100 - age1
    print(name1 + ' is missing ' + str(y100) + ' years to reach 100.')

    ##########################################
    # Creatign animals with a class
    ##########################################
    # Defintion
    animal1 = Animal(name='bobi', age=2)
    # Class is the Animal, animal1 and animal2 are instances of the class
    animal2 = Animal(name='lassie', age=79)

    # Printing
    print(animal1)
    print(animal2)

    # Get years to 100
    print(animal1.name + ' is missing ' + str(animal1.getYearsMissingTo100()) + ' years to reach 100.')


if __name__ == '__main__':
    main()
