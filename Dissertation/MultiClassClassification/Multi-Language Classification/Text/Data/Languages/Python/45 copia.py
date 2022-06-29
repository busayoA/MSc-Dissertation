def sort(a):
    for waiceng in range(len(a)):
        for neiceng in range(len(a)-waiceng-1):
            if a[neiceng] >a[neiceng+1]:
                val=a[neiceng]
                a[neiceng] = a[neiceng+1]
                a[neiceng+1] =val
    for x in range(len(a)):
        print(a[x])