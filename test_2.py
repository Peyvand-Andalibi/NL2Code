import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

val_bleu_all = [[0, 73], [1, 75], [2, 77], [3, 79]]
val_acc_all = [[0, 70], [1, 72], [2, 74], [3, 76]]
val_loss_all = [[0, 12], [1, 10], [2, 8], [3, 7]]

f = open("val_bleu_all.csv", mode='w+')
with f:
    write = csv.writer(f)
    write.writerows(val_bleu_all)
f.close()

f = open("val_acc_all.csv", mode='w+')
with f:
    write = csv.writer(f)
    write.writerows(val_acc_all)
f.close()

f = open("val_loss_all.csv", mode='w+')
with f:
    write = csv.writer(f)
    write.writerows(val_loss_all)
f.close()

val_bleu_all = np.transpose(val_bleu_all)
val_acc_all = np.transpose(val_acc_all)
val_loss_all = np.transpose(val_loss_all)