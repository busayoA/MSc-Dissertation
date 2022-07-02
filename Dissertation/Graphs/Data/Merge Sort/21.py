def merge (a, b, c=[]):
     if len(a) == 0:
         return c + b
     elif len(b) == 0:
         return c + a
     else:
         ah = a[0]
         bh = b[0]
         if ah <= bh:
             return merge (a[1:], b, c + [ah])
         else:
             return merge (a, b[1:], c + [bh])

def mergesort (a):
     l = len(a)
     if l < 2:
         return a
     else:
         m = l // 2   
         return merge(mergesort(a[0:m]), mergesort(a[m:]))