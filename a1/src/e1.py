import numpy as np

trainIn = np.genfromtxt('../data/train_in.csv', delimiter=',')
trainOut = np.genfromtxt('../data/train_out.csv', delimiter=',')

i = 0
meanSum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
meanOccurence = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cloud = [[],[],[],[],[],[],[],[],[],[]]

for imgToBeRecognized in np.nditer(trainOut.T):
  recognizedNumber = int(trainOut[i])
  meanSum[recognizedNumber] += trainIn[i]
  meanOccurence[recognizedNumber] += 1
  cloud[recognizedNumber].append(trainIn[i])
  i += 1

center = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
radius = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for b in range(0, 10):
  center[b] = meanSum[b] / meanOccurence[b]
  for img in cloud[b]:
    res = np.absolute(np.linalg.norm(center[b] - img))
    if res > radius[b]:
      radius[b] = res


totalDistance = np.zeros([10, 10])
for b in range(0, 10):
  for c in range(0, 10):
    totalDistance[b][c] += np.absolute(np.linalg.norm(center[b] - center[c]))

print("distances are ", totalDistance)
np.savetxt("distances.csv", totalDistance, delimiter=",")