import os
import sys

def check_mnist_dataset_exists(path_data='./'):
    flag_train_data = os.path.isfile(path_data + 'mnist/train_data.pt')
    flag_train_label = os.path.isfile(path_data + 'mnist/train_label.pt')
    flag_test_data = os.path.isfile(path_data + 'mnist/test_data.pt')
    flag_test_label = os.path.isfile(path_data + 'mnist/test_label.pt')
    if flag_train_data == False or flag_train_label == False or flag_test_data == False or flag_test_label == False:
        print('MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=True,
                                              download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=False,
                                             download=True, transform=transforms.ToTensor())
        train_data = torch.Tensor(60000, 28, 28)
        train_label = torch.LongTensor(60000)

        for idx, example in enumerate(trainset):
            train_data[idx] = example[0].squeeze()
            train_label[idx] = example[1]

        torch.save(train_data, path_data + 'mnist/train_data.pt')
        torch.save(train_label, path_data + 'mnist/train_label.pt')
        test_data = torch.Tensor(10000, 28, 28)
        test_label = torch.LongTensor(10000)

        for idx, example in enumerate(testset):
            test_data[idx] = example[0].squeeze()
            test_label[idx] = example[1]

        torch.save(test_data, path_data + 'mnist/test_data.pt')
        torch.save(test_label, path_data + 'mnist/test_label.pt')
    return path_data
