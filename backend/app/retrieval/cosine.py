import math

def dot_product(vector1, vector2):
    dot_product = 0
    for x,y in zip(vector1,vector2):
        dot_product += x * y
    return dot_product

def magnitude(vector):
    magnitude = 0
    for x in vector:
        magnitude += x**2
    return math.sqrt(magnitude)

def cosine_similarity(vector1,vector2):
   numerator =  dot_product(vector1,vector2)
   denominator = magnitude(vector1)*magnitude(vector2)
   if denominator == 0:
       return 0
   return numerator/denominator

