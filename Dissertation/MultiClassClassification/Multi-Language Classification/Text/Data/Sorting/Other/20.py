def insertSort(v):
      for j in range(1, len(v)): 
        key = v[j] 
        i = j-1
  
        while(i >= 0 and v[i] > key): 
            v[i+1] = v[i] 
            i -= 1
  
            yield v 
        
        v[i+1] = key 
        yield v 