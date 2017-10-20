class Vehicle:
   def __init__(self, number_of_wheels, type_of_tank, seating_capacity, maximum_velocity):
       self.number_of_wheels = number_of_wheels
       self.type_of_tank = type_of_tank
       self.seating_capacity = seating_capacity
       self.maximum_velocity = maximum_velocity
       
car = Vehicle(4,1,1,2)
print(car)