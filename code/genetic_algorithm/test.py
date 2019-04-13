from metrics import compare
from metrics import get_first_structural_distance
from metrics import get_second_structural_distance
import numpy as np
from create_population import create_population


np.random.seed(42)
L = create_population(4, 10)
np.random.seed(42)
R = create_population(4, 10)

for l in L:
	l.print_as_tree()
	print (l.get_str_representation())
	print ("|")

print ("if compare works, matrix should be id")

for l in L:
	for r in R:
		if compare(l, r):
			if L.index(l) != R.index(r):
				print ("algo thinks that")
				l.print_as_tree()
				print ("equals to " + "-" * 100)
				r.print_as_tree()
				print ("-------------\n" * 4)
			print ("1", end="")
		else:
			print (".", end="")
	print ()


print ("Max common subtree dist")

for l in L:
	for r in R:
		print ("%.2d" % (get_first_structural_distance(l, r),), end=" ")
	print ()

print ("Levenshtein str dist")

for l in L:
	for r in R:
		print ("%.2d" % (get_second_structural_distance(l, r),), end=" ")
	print ()

