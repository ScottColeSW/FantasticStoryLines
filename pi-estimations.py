
# The Chudnovsky algorithm was published by the Chudnovsky brothers in 1988 
# as a fast method for calculating the digits of π, 
# based on Ramanujan’s π formulae.
import decimal
import math

print("Python's default Pi value: ",math.pi)

def compute_pi(n):
  """
  This function calculates the value of pi to 'n' number of decimal places
  Args:
  n:   precision(Decimal places)
  Returns:
  pi:   the value of pi to n-decimal places
  """
  
  decimal.getcontext().prec = n + 3
  decimal.getcontext().Emax = 999999999
  
  C = 426880 * decimal.Decimal(10005).sqrt()
  K = decimal.Decimal(6)
  M = decimal.Decimal(1)
  X = decimal.Decimal(1)
  L = decimal.Decimal(13591409)
  S = L
  
  # For better precision, we calculate to n+3 and truncate the last two digits
  for i in range(1, n+3):
    M = decimal.Decimal(M* ((1728*i*i*i)-(2592*i*i)+(1104*i)-120)/(i*i*i))
    L = decimal.Decimal(545140134+L)
    X = decimal.Decimal(-262537412640768000*X)
    S += decimal.Decimal((M*L) / X)
    
  return str(C/S)[:-2] # Pi is C/S


n = int(input("Please type number: "))
print(compute_pi(n))

# This will write the calculated Pi to a file and this can be 
# used to compare to existing Pi calculations
# with open("pi_digits.txt", mode='w') as file:
#     file.write('Pi digits to ' + str(n) +' decimal place is:'\
#     +'\n' + compute_pi(n))