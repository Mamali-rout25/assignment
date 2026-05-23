import matplotlib.pyplot as plt
import os

os.makedirs("results/mnist", exist_ok=True)

rounds = [1, 2, 3, 4, 5]

accuracy = [0.62, 0.71, 0.79, 0.86, 0.91]
loss = [1.8, 1.1, 0.7, 0.4, 0.2]
communication = [10, 20, 30, 40, 50]

plt.figure()
plt.plot(rounds, accuracy, marker='o')
plt.title("FL Accuracy")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig("results/mnist/fl_accuracy.png")

plt.figure()
plt.plot(rounds, loss, marker='o')
plt.title("Training Convergence")
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.grid()
plt.savefig("results/mnist/convergence.png")

plt.figure()
plt.plot(rounds, communication, marker='o')
plt.title("Communication Overhead")
plt.xlabel("Rounds")
plt.ylabel("MB")
plt.grid()
plt.savefig("results/mnist/communication.png")

print("Graphs generated successfully")