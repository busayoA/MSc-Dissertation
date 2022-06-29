def mergeSort( lst ):
    if len( lst ) == 1:
        return lst
    else:
        half = len( lst ) // 2
        list1 = lst[:half]
        list2 = lst[half:]
        return merge( mergeSort( list1 ), mergeSort( list2 ))

def merge( list1, list2 ):
    newlst = []
    while len( list1 ) and len( list2 ):
        element = min( list1[0], list2[0] )
        newlst.append( element )
        if element == list1[0]:
            list1.remove( element )
        else:
            list2.remove( element )
    if len( list1 ):
        newlst.extend( list1 )
    elif len( list2 ):
        newlst.extend( list2 )
    return newlst