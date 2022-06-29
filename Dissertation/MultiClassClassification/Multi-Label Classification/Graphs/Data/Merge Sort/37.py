def  mergeSort(self, array) :
        if (len(array) > 1) :
            elementsInA1 = int(len(array) / 2)
            elementsInA2 = len(array) - elementsInA1
            arr1 = [0] * (elementsInA1)
            arr2 = [0] * (elementsInA2)
            i = 0
            while (i < elementsInA1) :
                arr1[i] = array[i]
                i += 1
            i = elementsInA1
            while (i < elementsInA1 + elementsInA2) :
                arr2[i - elementsInA1] = array[i]
                i += 1
            arr1 = self.mergeSort(arr1)
            arr2 = self.mergeSort(arr2)
            i = 0
            j = 0
            k = 0
            while (len(arr1) != j and len(arr2) != k) :
                if (arr1[j] < arr2[k]) :
                    array[i] = arr1[j]
                    i += 1
                    j += 1
                else :
                    array[i] = arr2[k]
                    i += 1
                    k += 1
            while (len(arr1) != j) :
                array[i] = arr1[j]
                i += 1
                j += 1
            while (len(arr2) != k) :
                array[i] = arr2[k]
                i += 1
                k += 1
        return array