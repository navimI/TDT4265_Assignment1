import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # find index of max value in the predictions
    predictions = model.forward(X)
    predictions = np.argmax(predictions, axis=1)

    # calculate accuracy
    accuracy = np.count_nonzero(np.equal(predictions, targets.argmax(axis=1))) / X.shape[0]

    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        #Froward pass
        output = self.model.forward(X_batch)
        #Backward pass
        self.model.backward(X_batch, output, Y_batch)
        #updating the parameters(gradient descent)
        self.model.w -= self.learning_rate * self.model.grad
        #return mean loss
        loss = cross_entropy_loss(Y_batch, output)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val

def print_landas(l2_lambdas, normalization, train_history_list, val_history_list):
    """
    Print the accuracy of the model on differents values of landa.
    Args:
        l2_lambdas (list): list of l2 regularization parameter
        train_history_list (list): list of train history
        val_history_list (list): list of validation history
    """
    for l, train_history, val_history in zip(l2_lambdas, train_history_list, val_history_list):
        plt.xlim([-100,5000])
        plt.ylim([0.7, .90])
        utils.plot_loss(train_history["accuracy"],
                        "Training Accuracy $\lambda$ = "+str(l))
        utils.plot_loss(val_history["accuracy"],
                        "Validation Accuracy $\lambda$ = "+str(l))
        plt.xlabel("Number of Training Steps")
        plt.ylabel("Accuracy")
        plt.legend()
    
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight
    plt.plot(l2_lambdas, normalization)
    plt.title("Plot of length of w vs $\lambda$")
    plt.xlabel("$\lambda$")
    plt.ylabel("$||w|^2$")
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()
        

def train_landas(learning_rate,batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,num_epochs):
    """
    Train the model with different values of l2 regularization parameter.
    Args:
        learning_rate (float): learning rate
        batch_size (int): batch size
        shuffle_dataset (bool): shuffle dataset
        X_train (np.ndarray): training images
        Y_train (np.ndarray): training labels
        X_val (np.ndarray): validation images
        Y_val (np.ndarray): validation labels
        num_epochs (int): number of epochs
    """

    l2_lambdas = [1, .1, .01, .001, 0]
    #Setting up the history lists
    normalization = []
    train_his_l = []
    val_his_l = []
    for l2 in l2_lambdas:
        # Initialize the model
        model = SoftmaxModel(l2)

        # Train model
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        # Adding the list of the trainer
        train_history, val_history = trainer.train(num_epochs)
        train_his_l.append(train_history)
        val_his_l.append(val_history)

        # Printing the results of the model

        print("Final Train Cross Entropy Loss:",
            cross_entropy_loss(Y_train, model.forward(X_train)))
        print("Final Validation Cross Entropy Loss:",
            cross_entropy_loss(Y_val, model.forward(X_val)))
        print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
        print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

        # Save the norm for plotting later
        normalization.append(np.sum(model.w*model.w))
    print_landas(l2_lambdas,normalization, train_his_l, val_his_l)

def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 500
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .8])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.

    modelWeight = np.hstack(model.w.T[:, :-1].reshape((-1, 28, 28)))
    model1Weight = np.hstack(model1.w.T[:, :-1].reshape((-1, 28, 28)))

    image4b = np.vstack((modelWeight, model1Weight))

    # Plotting of softmax weights (Task 4b)
    plt.imsave("task4b_softmax_weight.png", image4b, cmap="gray")
    plt.show()

    # Plotting of accuracy for difference values of lambdas (task 4c)
    # Task 4d - Plotting of the l2 norm for each weight
    train_landas(learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val, num_epochs)


    


if __name__ == "__main__":
    main()
