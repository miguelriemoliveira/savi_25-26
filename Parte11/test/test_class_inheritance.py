#!/usr/bin/env python3

class Animal:

    def __init__(self, weight, height):
        self.alive = True
        self.weight = weight
        self.height = height
        print("Animal __init_ called. Alive:", self.alive,
              "Weight:", self.weight, "Height:", self.height)


class Dog(Animal):

    def __init__(self, weight, height, name):

        super(Dog, self).__init__(weight, height)
        self.name = name

        print("Dog __init_ called. Name:", self.name)


# Run some code for creating an instance of Dog

dog1 = Dog(weight=10, height=0.5, name="Rex")
