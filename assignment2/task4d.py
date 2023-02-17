##TASK 4d
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    #first example without any improvement

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_orig, val_history_orig = trainer.train(num_epochs)


    neurons_per_layer = [64, 64, 64, 10]
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_doubled, val_history_doubled = trainer.train(num_epochs)

    neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_ten, val_history_ten = trainer.train(num_epochs)

    
    
    #plt.subplot(1, 2, 1)
    plt.title("Training and Validation loss")

    utils.plot_loss(train_history_orig["loss"],
                    "Training Task 2 Model", npoints_to_average=10)
    utils.plot_loss(val_history_orig["loss"], ("Validation Task 2 Model"))

    utils.plot_loss(
        train_history_doubled["loss"], "Training Task 2 Model - Improved weights", npoints_to_average=10)
    utils.plot_loss(val_history_doubled["loss"], ("Validation Task 2 Model - Double layer"))

    utils.plot_loss(
        train_history_ten["loss"], "Training Task 2 Model - Improved sigmoid", npoints_to_average=10)
    utils.plot_loss(val_history_ten["loss"], ("Validation Task 2 Model - Ten layers"))

    plt.ylim([0, .4])
    plt.xlabel("Training steps")
    plt.ylabel("Average Cross entropy Loss")
    plt.legend()
    #plt.savefig("task3a_loss.png")
    plt.show()
    #plt.subplot(1, 2, 2)
    plt.ylim([0.85, .95])
    utils.plot_loss(val_history_orig["accuracy"], "Task 2 Model")
    utils.plot_loss(train_history_orig["accuracy"], "Task 2 Model")
    utils.plot_loss(val_history_doubled["accuracy"], "Task 2 Model with improved weights")
    utils.plot_loss(train_history_doubled["accuracy"], ("Training Task 2 Model train with improved weights"))
     
    utils.plot_loss(val_history_ten["accuracy"], "Task 2 Model with improved sigmoid")
    utils.plot_loss(train_history_ten["accuracy"], ("Training Task 2 Model train with improved sigmoid"))
    
    plt.xlabel("Training steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    #plt.savefig("task3a_acc.png")
    plt.show()


if __name__ == "__main__":
    main()
