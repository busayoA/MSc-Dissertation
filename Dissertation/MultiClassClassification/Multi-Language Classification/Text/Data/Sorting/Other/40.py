def bubble_sort(s):
    n = len(s)
	for j in range(n - 1):
        for i in range(n - j - 1):
		    if s[i] > s[i + 1]:
				s[i], s[i + 1] = s[i + 1], s[i]
