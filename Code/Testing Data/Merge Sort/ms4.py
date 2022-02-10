# Source: https://gist.github.com/johnpena/817077/e4ac3d1600d30814d0472ece826322c0a76cbe0e 
def merge(left, right):
    result = []
    while len(left) > 0 or len(right) > 0:
        if len(left) > 0 and len(right) > 0:
            if left[0] <= right[0]:
                result.append(left.pop(0))
            else:
                result.append(right.pop(0))
        elif len(left) > 0:
            result.append(left.pop(0))
        elif len(right) > 0:
            result.append(right.pop(0))
    return result

def merge_sort(l):
    if len(l) <= 1:
        return l
    middle = len(l)/2
    left = merge_sort(l[:middle])
    right = merge_sort(l[middle:])
    return merge(left, right)