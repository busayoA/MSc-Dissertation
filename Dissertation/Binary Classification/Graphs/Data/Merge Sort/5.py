def mergeSort(elements,l,r):
    mid=(l+r)//2
    print(f"Divided from {l} to {r}")
    print(f"Left are:{elements[l:mid+1]}")
    print(f"Right are:{elements[mid+1:r+1]}")
    if l>=r:
        return elements
    else:
        mergeSort(elements,l,mid)
        mergeSort(elements,mid+1,r)
        merge(elements,l,r,mid)

def merge(list,l,r,mid):
    #print(list)
    d=list[l:mid+1]
    m=list[mid+1:r+1]
    #d indicates left list and m indicates right list
    #sort d
    for i in range(0,len(d)):
        for j in range(i,len(d)):
            if(d[i]>d[j]):
                d[i],d[j]=d[j],d[i]
    #sort m
    for i in range(0,len(m)):
        for j in range(i,len(m)):
            if(m[i]>m[j]):
                m[i],m[j]=m[j],m[i]
    #concatenate d and m
    c=d+m
    #sort the concatenated list
    for i in range(0,len(c)):
        for j in range(i,len(c)):
            if(c[i]>c[j]):
                c[i],c[j]=c[j],c[i]
    print(d)
    print(m)
    #print(c)
    #change the original list
    elements=c
    print(elements)
    # print(list[mid+1:r+1])