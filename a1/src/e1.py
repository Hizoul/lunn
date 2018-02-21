'''
Created on 10.02.2018

@author: Matthias Müller-Brockhausen & Oliver Scherf
'''
import numpy as np

def euclidianDistance(a, b):
    return np.absolute(np.linalg.norm(a - b))

def main():
    trainIn = np.genfromtxt('../data/train_in.csv', delimiter=',')
    trainOut = np.genfromtxt('../data/train_out.csv', delimiter=',')
    
    meanSum = [0] * 10
    meanOccurence =  [0] * 10
    cloud = [ [] for i in range(10) ]
    
    i = 0
    for _ in np.nditer(trainOut.T):
      recognizedNumber = int(trainOut[i])
      meanSum[recognizedNumber] += trainIn[i]
      meanOccurence[recognizedNumber] += 1
      cloud[recognizedNumber].append(trainIn[i])
      i += 1
    
    center = [0] * 10
    radius = [0] * 10
    for b in range(0, 10):
      center[b] = meanSum[b] / meanOccurence[b]
      for img in cloud[b]:
        res = euclidianDistance(center[b], img)
        if res > radius[b]:
          radius[b] = res
    
    
    totalDistance = np.zeros([10, 10])
    for b in range(0, 10):
      for c in range(0, 10):
        totalDistance[b][c] += euclidianDistance(center[b], center[c])
    
    print("distances are")
    print(totalDistance)
    np.savetxt("distances.csv", totalDistance, delimiter=",")

if __name__ == "__main__":
    main()