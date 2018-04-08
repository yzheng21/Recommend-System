# import matplotlib.pyplot as plt
# from math import*
#
# db = []
# pos, neg = 0, 0
# with open('data.txt') as f:
#     for line in f.readlines():
#         nonclk, clk, score = line.strip().split('\t')
#         db.append([float(score),int(nonclk),int(clk)])
#         pos += int(clk)
#         neg += int(nonclk)
#
# db = sorted(db, key=lambda f: f[0],reverse=True)
# xy_arr = []
# tp, fp = 0., 0.
# for i in range(len(db)):
#     tp += db[i][2]
#     fp += db[i][1]
#     xy_arr.append([tp/pos,fp/neg])
#
# auc = 0.
# prev_x = 0
# for x,y in xy_arr:
#     if x != prev_x:
#         auc += (x-prev_x)*y
#         prev_x = x
#
# print("accuracy: %s" % auc)
#
# x = [v[1] for v in xy_arr]
# y = [v[0] for v in xy_arr]
# plt.title("ROC curve of %s (AUC = %.4f)" % ('svm',auc))
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.plot(x, y)# use pylab to plot x and y
# plt.show()

from sklearn.metrics import roc_curve
import numpy as np
y = np.array([1, 0, 1, 1, 0, 0])
scores = np.array([0.99, 0.96, 0.90, 0.87, 0.85, 0.70])
fpr, tpr, thresholds = roc_curve(y, scores)
print(tpr
      )