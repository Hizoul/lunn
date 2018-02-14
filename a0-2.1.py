import numpy as np
import matplotlib.pyplot as plt

def y(x):
  return 0.5 + 0.4 * np.sin(np.pi * x)

n = [9, 15, 100]
i = 1
def makeSet(nVal):
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
  plt.plot(xPoints, correspondingYVals, color="blue")
  fig.savefig("noisy" + str(nVal) + ".png")

makeSet(100)