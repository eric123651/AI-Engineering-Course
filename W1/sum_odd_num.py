def generate_number_list(n):
    if n<0 :
      raise ValueError("404")
    number_list = []
    for i in range (1, n+1):
       number_list.append(i)
    return number_list 
def sum_odd_numbers(numbers):
    return sum(num for num in numbers if num %2 !=0)

n = 7
numbers = generate_number_list(n)

print (generate_number_list(n))
print (sum_odd_numbers(numbers))
