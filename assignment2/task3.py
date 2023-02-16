import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    #
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
    train_history, val_history = trainer.train(num_epochs)


    #model trained with improved weights
    use_improved_weight_init = True
    use_improved_sigmoid = False
    use_momentum = False

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

    train_history_weights, val_history_weights = trainer_weights.train(num_epochs)
    
    #model trained with sigmoid
    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = False

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
    
    train_history_sigmoid, val_history_sigmoid = trainer_sigmoid.train(num_epochs)
    
    #model trained with momentum
    learning_rate = .02 # lr for momentum
    use_improved_weight_init = True
    use_improved_sigmoid = True
    use_momentum = True

    model_momentum = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_momentum = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_momentum, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    train_history_momentum, val_history_momentum = trainer_momentum.train(num_epochs)
    
    
    #plt.subplot(1, 2, 1)
    plt.title("Training and Validation loss")

    utils.plot_loss(train_history["loss"],
                    "Training Task 2 Model", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], ("Validation Task 2 Model"))


    utils.plot_loss(
        train_history_weights["loss"], "Training Task 2 Model - Improved weights", npoints_to_average=10)
    utils.plot_loss(val_history_weights["loss"], ("Validation Task 2 Model - Improved weights"))

    utils.plot_loss(
        train_history_sigmoid["loss"], "Training Task 2 Model - Improved sigmoid", npoints_to_average=10)
    utils.plot_loss(val_history_sigmoid["loss"], ("Validation Task 2 Model - Improved sigmoid"))
     
    utils.plot_loss(
        train_history_momentum["loss"], "Training Task 2 Model - Improved momentum", npoints_to_average=10)
    utils.plot_loss(val_history_momentum["loss"], ("Validation Task 2 Model - Improved momentum"))
    
    plt.ylim([0, .4])
    plt.xlabel("Training steps")
    plt.ylabel("Average Cross entropy Loss")
    plt.legend()
    #plt.savefig("task3a_loss.png")
    plt.show()
    #plt.subplot(1, 2, 2)
    plt.ylim([0.85, .95])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(train_history["accuracy"], "Task 2 Model")
    utils.plot_loss(val_history_weights["accuracy"], "Task 2 Model with improved weights")
    utils.plot_loss(train_history_weights["accuracy"], ("Training Task 2 Model train with improved weights"))
     
    utils.plot_loss(val_history_sigmoid["accuracy"], "Task 2 Model with improved sigmoid")
    utils.plot_loss(train_history_sigmoid["accuracy"], ("Training Task 2 Model train with improved sigmoid"))
    
    utils.plot_loss(val_history_momentum["accuracy"], "Task 2 Model with momentum")
    utils.plot_loss(train_history_momentum["accuracy"], ("Training Task 2 Model train with momentum"))
    
    plt.xlabel("Training steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    #plt.savefig("task3a_acc.png")
    plt.show()


if __name__ == "__main__":
    main()
