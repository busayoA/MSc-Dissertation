from random import randint
def quick_sort(list):
     if len(list) > 1:
         num = list[randint(0, len(list) - 1)]
         low = [y for y in list if y < num]
         equal = [y for y in list if y == num]
         big = [y for y in list if y > num]
         d = quick_sort(low) + equal + quick_sort(big)
     return d 