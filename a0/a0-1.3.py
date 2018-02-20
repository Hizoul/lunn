import numpy
import matplotlib.pyplot as plt


Mx12 = numpy.array([[0, 0, 1, 1], [0, 1, 0, 1]])


axisPos = [0, 0, 1, 1]

Mand = numpy.array([0, 0, 0, 1])
Mor = numpy.array([0, 1, 1, 0])

rang = numpy.arange(0, 1, 0.1)
xx, yy = numpy.meshgrid(rang, rang, sparse=True)
z = numpy.sin(xx**2 + yy ** 2)


f1 = plt.figure(1)
plt.plot(0, 0, marker="o", color="green", label="p")
plt.plot(0, 1, marker="o", color="green", label="p")
plt.plot(1, 0, marker="o", color="green", label="p")
plt.plot(1, 1, marker="o", color="green", label="p")

plt.plot(Mand, axisPos, color="blue")
plt.contour(rang, rang, z)
plt.axis([-0.05, 1.05, -0.05, 1.05])
# plt.axis([-0.05, 1.05, -0.05, 1.05])
f1.savefig("desicionboundary.png")
