list = [0,2,100,0,120,11]

for countOne in range(0, len(list) - 1 ):
  max = list[countOne]
  maxIndex = countOne
  for countTwo in range(countOne + 1,len(list)):
    if list[countTwo] > max:
      max = list[countTwo]
      maxIndex = countTwo
  if maxIndex != countOne:
    temp = list[countOne]
    list[countOne] = max
    list[maxIndex] = temp
print(list)