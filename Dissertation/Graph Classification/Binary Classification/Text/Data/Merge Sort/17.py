def sort(myList, lBound, uBound):
    if uBound - lBound <= 1:
        return
    middle, partition = lBound + (uBound - lBound)//2, []
    sort(myList, lBound, middle)
    sort(myList, middle, uBound)
    merge(myList, lBound, middle, uBound, partition)

def merge(myList, lBound, middle, uBound, partition):
    start, end, index = lBound, middle, 0
    while start < middle and end < uBound:
        if myList[start] <= myList[end]:
            partition.append(myList[start])
            start += 1
        else:
            partition.append(myList[end])
            end += 1
        index += 1
        
    while start < middle:
        partition.append(myList[start])
        start += 1
        index +=1
    
    while end < uBound:
        partition.append(myList[end])
        end += 1
        index +=1
    
    index = 0
    for i in range(lBound, uBound, 1):
        myList[i] = partition[index]
        index += 1
    return partition