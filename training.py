from torch.autograd import Variable
import torch
def train(num_epochs,train_loader,test_loader,model,error,optimizer,scheduler,device):
    count = 0
    # loss and accuracy
    loss_list = []
    iteration_list = []
    accuracy_list = []

    #  classwise accuracy
    predictions_list = []
    labels_list = []

    for epoch in range(num_epochs):
        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            train = Variable(images.view(100, 1, 28, 28))
            # (batch size, channel, width, height)
            labels = Variable(labels)

            # Forward pass
            outputs = model(train)
            loss = error(outputs, labels)

            #Add Scheduler
            scheduler.step()

            # Initializing a gradient as 0
            optimizer.zero_grad()

            # Propagating the error backward
            loss.backward()

            # Optimizing the parameters
            optimizer.step()

            count += 1

            # Testing the model

            if not (count % 50):
                total = 0
                correct = 0

                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    labels_list.append(labels)

                    test = Variable(images.view(100, 1, 28, 28))

                    outputs = model(test)

                    predictions = torch.max(outputs, 1)[1].to(device)
                    predictions_list.append(predictions)
                    correct += (predictions == labels).sum()

                    total += len(labels)

                accuracy = correct * 100 / total
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

            if not (count % 100):
                print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))
    return loss_list,iteration_list,accuracy_list,predictions_list,labels_list