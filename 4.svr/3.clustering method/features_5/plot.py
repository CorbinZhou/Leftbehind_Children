import numpy as np 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

data = []
for i in range(10):
	data.append(pd.read_csv('results/KMdata'+str(i+1)+'.txt'))

	# plt.plot(data[i]['generation'],data[i]['train'])
	# plt.hold(True)
	# plt.plot(data[i]['generation'],data[i]['validation'])
	# plt.hold(True)
	# plt.plot(data[i]['generation'],data[i]['test'])
	# plt.savefig('fig/figure_2d_'+str(i+1)+'.jpg')
	# # plt.show()
	# plt.clf()

	plt.scatter(data[i]['validation'],data[i]['test'],s=2)
	plt.hold(True)
	# plt.savefig('fig/figure_2d2_'+str(i+1)+'.jpg')
	# plt.clf()

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(data[i]['train'],data[i]['validation'],data[i]['test'])
	# ax.set_xlabel('train')
	# ax.set_ylabel('validation')
	# ax.set_zlabel('test')

	# plt.savefig('fig/figure_3d_'+str(i+1)+'.jpg')
	# # plt.show()
	# plt.clf() 

plt.xlim((-0.2,0.9))
plt.savefig('fig/figure.jpg')