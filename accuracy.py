import torch
import output
from torch.autograd import Variable
def show(test_loader,model,device):
    class_correct = [0. for _ in range(10)]
    total_correct = [0. for _ in range(10)]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test = Variable(images)
            outputs = model(test)
            predicted = torch.max(outputs, 1)[1]
            c = (predicted == labels).squeeze()

            for i in range(100):
                label = labels[i]
                class_correct[label] += c[i].item()
                total_correct[label] += 1

    for i in range(10):
        print("Accuracy of {}: {:.2f}%".format(output.output_label(i), class_correct[i] * 100 / total_correct[i]))