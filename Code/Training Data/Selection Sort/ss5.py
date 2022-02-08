# Source: https://www.geeksforgeeks.org/python-program-for-selection-sort/

def sort(myList):
    for i in range(len(myList)):
        min_idx = i
        for j in range(i+1, len(myList)):
            if myList[min_idx] > myList[j]:
                min_idx = j  
        myList[i], myList[min_idx] = myList[min_idx], myList[i]