def mergeSort(arr):
    
    #calculate separtor med
    if len(arr) > 1:
        mid = len(arr)//2

        left = arr[:mid]
        right = arr[mid:]
        mergeSort(left)
        mergeSort(right)


        #iterators for the 2 separated arrays
        i = 0
        j = 0

        #iterator for the combined array
        k = 0

        while i<len(left) and j<len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[j] = right[j]
                j += 1
            k +=1

        #for the remianing
        while i<len(left):
            arr[k] = left[i]
            i +=1
            k +=1
        while j<len(right):
            arr[k] = right[j]

            j +=1
            k +=1