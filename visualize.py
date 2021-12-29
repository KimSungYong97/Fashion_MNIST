import matplotlib.pyplot as plt
import torch
def plot(iteration_list,loss_list,accuracy_list):
    plt.plot(torch.tensor(iteration_list).cpu(), torch.tensor(loss_list).cpu())
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.show()

    plt.plot(torch.tensor(iteration_list).cpu(), torch.tensor(accuracy_list).cpu())
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.show()