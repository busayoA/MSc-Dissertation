def quicksort1(l):
    size = len(l)
    # stop condition for reqursive calls
    if size<=1:
        return l
    # buckets for lesser and greater numbers
    lesser = []
    greater = []

    # find pivot element and remove it from the list 
    index = int(size/2)
    pivot = l[index]
    # it is okay to remove some other element that has same name with our element
    l.remove(pivot)
    #separate elements to greater and lesser buckets
    for i in l:
        if i >= pivot:
            greater.append(i)
        else:
            lesser.append(i)
    # call quicksort on lesser and greater buckets and concatenate resulted arrays
    lesser =  quicksort1(lesser)
    lesser.append(pivot)
    return lesser + quicksort1(greater)
    #return quicksort1(lesser)+[pivot]+quicksort1(greater)