def  mergesort(self, left,  right) :
        result =  []
        leftSize = left.size()
        rightSize = right.size()
        if (leftSize == 1 and rightSize == 1) :
            leftNum = left.remove(0)
            rightNum = right.remove(0)
            if (leftNum < rightNum) :
                result.add(leftNum)
                result.add(rightNum)
            else :
                result.add(rightNum)
                result.add(leftNum)
        newLeft = self.mergesort(left.subList(0,int(leftSize / 2)), left.subList(int(leftSize / 2),leftSize))
        newRight = self.mergesort(right.subList(0,int(leftSize / 2)), right.subList(int(leftSize / 2),leftSize))
        while (newLeft.size() > 0 or newRight.size() > 0) :
            if (newLeft.get(0) < newRight.get(0)) :
                result.add(newLeft.remove(0))
            else :
                result.add(newRight.remove(0))
        if (newLeft.size() != 0) :
            while (newLeft.size() > 0) :
                result.add(newLeft.remove(0))
        else :
            while (newRight.size() > 0) :
                result.add(newRight.remove(0))
        return result