def doQuickSort( leftBorder,  rightBorder,  sortArray,  metaDataArray) :
        i = 0
        j = 0
        x = 0.0
        swapTempDouble = 0.0
        swapTempString = None
        i = leftBorder
        j = rightBorder
        x = sortArray[int((leftBorder + rightBorder) / 2)]
        while True :
            while (sortArray[i] > x) :
                i += 1
            while (x > sortArray[j]) :
                j -= 1
            if (i <= j) :
                swapTempDouble = sortArray[i]
                sortArray[i] = sortArray[j]
                sortArray[j] = swapTempDouble
                swapTempString = metaDataArray[i]
                metaDataArray[i] = metaDataArray[j]
                metaDataArray[j] = swapTempString
                i += 1
                j -= 1
            if((not (i > j)) == False) :
                    break
        if (leftBorder < j) :
            doQuickSort(leftBorder, j, sortArray, metaDataArray)
        if (i < rightBorder) :
            doQuickSort(i, rightBorder, sortArray, metaDataArray)