
arr = []
while True:
    n = input("Enter number or 'end' to finish: ")
    if n == "end":
        break
    else:
        n = int(n)
    arr.append(n)

arr.sort()
l = len(arr)
print(l)
if l % 2 == 0:
    med = (arr[l//2] + arr[(l//2)-1]) / 2
else:
    med = arr[l//2]

print("Median is: ", med)