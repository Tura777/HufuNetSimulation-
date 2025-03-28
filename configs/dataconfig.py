import torchvision
from torchvision import transforms

class Database:
    """
    This class provides utility methods to prepare datasets and their corresponding
    data transformations for training and testing deep learning models.

    Supported datasets:
    - MNIST
    - CIFAR10
    """

    @staticmethod
    def get_transforms(database):
        """
        Returns the appropriate torchvision transforms for the given dataset.

        Parameters:
            database (str): Name of the dataset ("mnist" or "cifar10").

        Returns:
            (transform_train, transform_test): Tuple of transformation functions
            to be applied to training and testing data respectively.

        Raises:
            Exception: If the dataset name is not recognized.
        """

        if database == "mnist":
            # Simple transforms for MNIST to convert to tensor
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        elif database == "cifar10":
            # Data augmentation for training, normalization for both training and testing
            transform_train = transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            raise Exception("database doesn't exist")
        return transform_train, transform_test

    @staticmethod
    def get_dataset(database, transform_train, transform_test):
        """
        Loads the specified dataset with the given transformations.

        Parameters:
            database (str): Name of the dataset ("mnist" or "cifar10").
            transform_train: Transformations to apply to the training set.
            transform_test: Transformations to apply to the testing set.

        Returns:
            (train_set, test_set): Tuple of training and testing datasets.

        Raises:
            Exception: If the dataset name is not recognized.
        """
        if database == "mnist":
            train_set = torchvision.datasets.MNIST(
                root='./data', train=True,
                transform=transform_train, download=True
            )
            test_set = torchvision.datasets.MNIST(
                root='./data', train=False,
                transform=transform_test
            )
        elif database == "cifar10":
            train_set = torchvision.datasets.CIFAR10(
                root='./data', train=True,
                transform=transform_train, download=True
            )
            test_set = torchvision.datasets.CIFAR10(
                root='./data', train=False,
                transform=transform_test
            )
        else:
            raise Exception("Unknown Database")
        return train_set, test_set

    @staticmethod
    def get_datasets(database):
        """

        Convenience method to get both training and testing datasets
        with their associated transforms.

        Parameters:
            database (str): Name of the dataset ("mnist" or "cifar10").

        Returns:
            (train_set, test_set): Tuple of training and testing datasets.
        """
        transform_train, transform_test = Database.get_transforms(database)
        return Database.get_dataset(database, transform_train, transform_test)


