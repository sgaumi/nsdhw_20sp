import pytest
import unittest
import numpy.random as rd
import time 
from Matrix import Matrix, multiply_naive

c=Matrix(2,2)
c[0,0]=3
a=c[0,0]
print(a)


m = rd.randint(300, 500)
n = rd.randint(100, 500)
k = rd.randint(300, 500)
m=10
n=10
k=10
print(" size m={}, k={}, n={} ".format(m, k, n))
d = Matrix(rd.random((m, k)))
e = Matrix(rd.random((k, n)))
#d=Matrix(m,k)
#e=Matrix(k,n)
#for i in range(0,m):
#	for j in range(0,k):
#		d[i,j]=1
#for i in range(0,k):
#	for j in range(0,n):
#		e[i,j]=1
print(d.nrow)
print(e.ncol)
#ans_nav = Matrix(m,n)
ans_nav = multiply_naive(d, e)
        #ans_mkl = multiply_mkl(a, b)

#assertEqual(ans_nav.nrow, m)
#assertEqual(ans_nav.ncol, n)
