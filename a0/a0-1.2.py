import numpy
import matplotlib.pyplot as plt


Mx12 = numpy.array([[0, 0, 1, 1], [0, 1, 0, 1]])


axisPos = [0, 0, 1, 1]

Mand = numpy.array([0, 0, 0, 1])
Mor = numpy.array([0, 1, 1, 0])


weights = numpy.random.randn(2)
bias = numpy.random.randn(2)
print("weights are", weights)

widehatY = Mx12 * weights + bias
print("got widehaty", widehatY)

heavisided = numpy.heaviside(widehatY, 1)


f1 = plt.figure(1)
plt.plot(0, 0, marker="o", color="green", label="p")
plt.plot(0, 1, marker="o", color="green", label="p")
plt.plot(1, 0, marker="o", color="green", label="p")
plt.plot(1, 1, marker="o", color="green", label="p")

plt.plot(heavisided, color="blue")

# plt.axis([-0.05, 1.05, -0.05, 1.05])
f1.savefig("heaviside.png")
