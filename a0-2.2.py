import numpy as np
import matplotlib.pyplot as plt

def y(x):
  return 0.5 + 0.4 * np.sin(np.pi * x)

n = [9, 15, 100]
degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
i = 1
def makeSet(nVal, degree):
  global i
  print("using nval ", nVal)
  xPoints = np.random.uniform(0, 1, nVal)
  xPoints.sort()
  yContain = np.random.normal(0, 0.05)
  correspondingYVals = []
  for x in xPoints:
    correspondingYVals.append(y(x) * yContain)
  fig = plt.figure(i)
  i = i + 1
  polyfitted = np.polyfit(xPoints, correspondingYVals, degree)
  print("polyfitted is", polyfitted)
  polyvalued = np.polyval(polyfitted, xPoints)
  plt.plot(xPoints, correspondingYVals, color="blue")
  plt.plot(xPoints, polyvalued, color="red")
  fig.savefig("a22/noisyfitted" + str(nVal) + "d" + str(degree) + ".png")

for nVal in n:
  for degree in degrees:
    makeSet(nVal, degree)