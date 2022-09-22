a = int(input("Enter first integer: "))
b = int(input("Enter second integer: "))

if a==b:
    print(True)
elif (a + b) == 5:
    print(True)
elif abs(a - b) == 5:
    print(True)
else:
    print(False)
