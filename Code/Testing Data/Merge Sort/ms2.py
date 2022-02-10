# Source: https://gist.github.com/YuheiNakasaka/2291821

def merge(l,r):
	result = []
	while l != [] and r != []:
		if l[0] < r[0]:
			result.append(l.pop(0))
		elif l[0] >= r[0]:
			result.append(r.pop(0))
	return result + l + r

def div(lst):
	if len(lst) == 1:
		return lst
	mid = len(lst)/2
	l = div(lst[0:mid])
	r = div(lst[mid:len(lst)])
	return merge(l,r)