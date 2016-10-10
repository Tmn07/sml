#coding=utf-8
from config import *


class c2(object):
	"""docstring for c2"""
	def __init__(self):
		#将b并入w
		self.w = [0,0,0]
		self.n = 1
		
	def func(self,xi):
		re = sum(map(lambda (a,b):a*b, zip(xi,self.w)))
		return re

	def func2(self,num,xi):
		dw = [0]*len(self.w)
		for i in xrange(len(xi)):
			dw[i] = num*xi[i]
		return dw

	def loop(self,x,y):
		"""
		"""
		while 1:
			flag = 0
			for i in xrange(len(x)):
				# print(x[i])
				re = self.func(x[i])
				# print re
				if re * y[i] > 0:
					print('right')
				else:
					print('wrong')
					print(i),
					self.w = map(lambda (a,b):a+b, zip(self.w,self.func2(self.n * y[i], x[i])))
					flag = 1
					print(self.w)
			if flag == 0:
				break

	def main(self):
		self.loop(x,y)
		print(self.w)

if __name__ == '__main__':
	c = c2()
	c.main()
	# print(b)


