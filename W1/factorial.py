def factorial(n):
    if n < 0:
        raise ValueError("n value can't lower than 0")
    if n > 1:
        for i in range (1, n+1):
            result *= i
    return result 
print (factorial(6))