import csv
import matplotlib.pyplot as plt
import numpy as np


train_bleu_all = [[0, 75], [1, 77], [2, 79], [3, 81]]
train_acc_all = [[0, 72], [1, 74], [2, 76], [3, 78]]
train_loss_all = [[0, 10], [1, 8], [2, 7], [3, 6]]
val_bleu_all = [[0, 73], [1, 75], [2, 77], [3, 79]]
val_acc_all = [[0, 70], [1, 72], [2, 74], [3, 76]]
val_loss_all = [[0, 12], [1, 10], [2, 8], [3, 7]]


f = open("train_bleu_all.csv", mode='w+')
with f:
    write = csv.writer(f)
    write.writerows(train_bleu_all)
f.close()

f = open("train_acc_all.csv", mode='w+')
with f:
    write = csv.writer(f)
    write.writerows(train_acc_all)
f.close()

f = open("train_loss_all.csv", mode='w+')
with f:
    write = csv.writer(f)
    write.writerows(train_loss_all)
f.close()

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

train_bleu_all = np.transpose(train_bleu_all)
train_acc_all = np.transpose(train_acc_all)
train_loss_all = np.transpose(train_loss_all)
val_bleu_all = np.transpose(val_bleu_all)
val_acc_all = np.transpose(val_acc_all)
val_loss_all = np.transpose(val_loss_all)

plt.figure(1)
plt.plot(train_bleu_all[0], train_bleu_all[1], 'r', label="Training Bleu")
plt.plot(val_bleu_all[0], val_bleu_all[1], 'b', label="Validation Bleu")
plt.title("Training and Validation Bleu")
plt.xlabel("Epochs")
plt.ylabel("Bleu")
plt.legend()
plt.savefig("Bleu_diagram.png")
# plt.show()

plt.figure(2)
plt.plot(train_acc_all[0], train_acc_all[1], 'r', label="Training Accuracy")
plt.plot(val_acc_all[0], val_acc_all[1], 'b', label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Accuracy_diagram.png")
# plt.show()

plt.figure(3)
plt.plot(train_loss_all[0], train_loss_all[1], 'r', label="Training Loss")
plt.plot(val_loss_all[0], val_loss_all[1], 'b', label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("Loss_diagram.png")
# plt.show()
