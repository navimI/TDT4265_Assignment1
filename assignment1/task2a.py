##TASK 2a
import array
import numpy as np
import math
import utils
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] in the range (-1, 1)
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    X=X/127.5 - 1
    bSize=X.shape[0]
    #print(bSize)
    #X=np.reshape(X, (bSize, 785), order='C')
    Y=np.concatenate((X,np.zeros((bSize,1))), axis=1)
    for value in range(bSize):
      Y[value][784]=1
    #print (Y[1][151])
    #print ("--------------------")
    #print (Y.shape[1])
    return Y


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, 1]
        outputs: outputs of model of shape: [batch size, 1]
    Returns:
        Cross entropy error (float)
    """
    # TODO implement this function (Task 2a)
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    bSize = targets.shape[0]
    tot=0
    for entry in range(bSize):
      tot+=(targets[entry]*math.log(outputs[entry]) + (1-targets[entry])*math.log(1 - outputs[entry]))*-1
    return tot/bSize


class BinaryModel:

    def __init__(self):
        # Define number of input nodes
        #self.I = None
        self.I = 785
        self.w = np.zeros((self.I, 1))
        self.grad = 0

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, 1]
        """
        # TODO implement this function (Task 2a)
        bSize=X.shape[0]
        nPixel=X.shape[1]
        wtx=X@self.w
        #wt=self.w.transpose()
        #wtx=np.dot(self.w.transpose(), X)
        y=1/(1+np.exp(-wtx))
        return y

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, 1]
            targets: labels/targets of each image of shape: [batch size, 1]
        """
        # TODO implement this function (Task 2a)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        self.grad = np.zeros_like(self.w)
        assert self.grad.shape == self.w.shape,\
            f"Grad shape: {self.grad.shape}, w: {self.w.shape}"
        bSize=targets.shape[0]
        #bSize=X.shape[0]
        #nPixel=X.shape[1]
        #for enume in range(bSize):
        #    for pixel in range(nPixel):
        #        X[enume][pixel]=-1*(targets[enume][0] - outputs[enume][0])*X[enume][pixel]
        self.grad = -1*X.transpose().dot(targets - outputs)/bSize #Maybe to recheck
        

    def zero_grad(self) -> None:
        self.grad = None


def gradient_approximation_test(model: BinaryModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    w_orig = np.random.normal(
        loc=0, scale=1/model.w.shape[0]**2, size=model.w.shape)
    epsilon = 1e-3
    for i in range(w_orig.shape[0]):
        model.w = w_orig.copy()
        orig = w_orig[i].copy()
        model.w[i] = orig + epsilon
        logits = model.forward(X)
        cost1 = cross_entropy_loss(Y, logits)
        model.w[i] = orig - epsilon
        logits = model.forward(X)
        cost2 = cross_entropy_loss(Y, logits)
        gradient_approximation = (cost1 - cost2) / (2 * epsilon)
        model.w[i] = orig
        # Actual gradient
        logits = model.forward(X)
        model.backward(X, logits, Y)
        difference = gradient_approximation - model.grad[i, 0]
        assert abs(difference) <= epsilon**2,\
            f"Calculated gradient is incorrect. " \
            f"Approximation: {gradient_approximation}, actual gradient: {model.grad[i,0]}\n" \
            f"If this test fails there could be errors in your cross entropy loss function, " \
            f"forward function or backward function"


def main():
    category1, category2 = 2, 3
    X_train, Y_train, *_ = utils.load_binary_dataset(category1, category2)
    X_train = pre_process_images(X_train)
    assert X_train.max(
    ) <= 1.0, f"The images (X_train) should be normalized to the range [-1, 1]"
    assert X_train.min() < 0 and X_train.min() >= - \
        1, f"The images (X_train) should be normalized to the range [-1, 1]"
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    # Simple test for forward pass. Note that this does not cover all errors!
    model = BinaryModel()
    logits = model.forward(X_train)
    np.testing.assert_almost_equal(logits.mean(), .5, err_msg="Since the weights are all 0's, the sigmoid activation should be 0.5")

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for i in range(2):
        gradient_approximation_test(model, X_train, Y_train)
        model.w = np.random.randn(*model.w.shape)


if __name__ == "__main__":
    main()
    
    
## NOTES
#https://mlnotebook.github.io/post/nn-in-python/#forwardpass
#https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html
#https://www.geeksforgeeks.org/ml-logistic-regression-using-python/
