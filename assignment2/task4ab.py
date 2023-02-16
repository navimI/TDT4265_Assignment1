import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    
    batch_size = 32
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    #64 neurons
    neurons_per_layer = [64, 10]

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
    train_history_64, val_history_64 = trainer.train(num_epochs)


    #model trained with 32 layers

    neurons_per_layer = [32, 10]


    model_weights = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_weights = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_weights, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    train_history_32, val_history_32 = trainer_weights.train(num_epochs)
    
    #model trained 128
    neurons_per_layer = [128, 10]

    model_sigmoid = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_sigmoid = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_sigmoid, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    
    train_history_128, val_history_128 = trainer_sigmoid.train(num_epochs)
    
    
    #plt.subplot(1, 2, 1)
    plt.title("Training and Validation loss")

    utils.plot_loss(train_history_64["loss"],
                    "Training Task 2 Model 64 units", npoints_to_average=10)
    utils.plot_loss(val_history_64["loss"], ("Validation Task 2 Model 64 units"))


    utils.plot_loss(
        train_history_32["loss"], "Training Task 2 Model 32 units", npoints_to_average=10)
    utils.plot_loss(val_history_32["loss"], ("Validation Task 2 Model 32 units"))

    utils.plot_loss(
        train_history_128["loss"], "Training Task 2 Model 128 units", npoints_to_average=10)
    utils.plot_loss(val_history_128["loss"], ("Validation Task 2 Model 128 units"))
    
    plt.ylim([0, .4])
    plt.xlabel("Training steps")
    plt.ylabel("Average Cross entropy Loss")
    plt.legend()
    #plt.savefig("task3a_loss.png")
    plt.show()
    #plt.subplot(1, 2, 2)
    plt.ylim([0.85, .95])
    utils.plot_loss(val_history_64["accuracy"], "Task 2 Model 64 units")
    utils.plot_loss(train_history_64["accuracy"], "Task 2 Model 64 units")
    utils.plot_loss(val_history_32["accuracy"], "Task 2 Model 32 units")
    utils.plot_loss(train_history_32["accuracy"], ("Training Task 2 Model 32 units"))
     
    utils.plot_loss(val_history_128["accuracy"], "Task 2 Model 128 units")
    utils.plot_loss(train_history_128["accuracy"], ("Training Task 2 Model 128 units"))
    
    plt.xlabel("Training steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    #plt.savefig("task3a_acc.png")
    plt.show()


if __name__ == "__main__":
    main()
