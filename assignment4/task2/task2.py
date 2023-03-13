import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE

    # Compute intersection
    assert (prediction_box.shape[0] == gt_box.shape[0])
    ixmin=max(prediction_box[0],gt_box[0])
    iymin=max(prediction_box[1],gt_box[1])
    ixmax=min(prediction_box[2],gt_box[2])
    iymax=min(prediction_box[3],gt_box[3])
    intersection=0
    if (ixmax > ixmin and iymax > iymin):
      intersection=(ixmax-ixmin)*(iymax-iymin)
    # Compute union
    pbox_area=(prediction_box[2]-prediction_box[0])*(prediction_box[3]-prediction_box[1])
    gtbox_area=(gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
    union = pbox_area + gtbox_area - intersection
    iou = intersection/union
    #END OF YOUR CODE

    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    # YOUR CODE HERE
    if num_tp + num_fp == 0:
      return 1
    else:
      precision=num_tp/(num_tp+num_fp)
      return precision
    #END OF YOUR CODE

def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    # YOUR CODE HERE
    if num_tp + num_fp == 0:
      return 0
    else:
      recall=num_tp/(num_tp+num_fn)
      return recall
    #END OF YOUR CODE

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x >= pivot]
    right = [x for x in arr if x < pivot]
    return quicksort(left) + quicksort(right)

def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # YOUR CODE HERE

    # Find all possible matches with a IoU >= iou threshold
    #assert(len(prediction_boxes) == len(gt_boxes))
    matches=[]
    for pn in range(prediction_boxes.shape[0]):
    #for pbox in prediction_boxes:
      pbox=prediction_boxes[pn]
      for gn in range(gt_boxes.shape[0]):
        gbox=gt_boxes[gn]
        #for gn in len(gt_boxes):
        iou=calculate_iou(pbox, gbox)
        if iou >= iou_threshold:
          #print(len(matches))
          matches.append([iou, pn, gn, pbox, gbox])

    # Sort all matches on IoU in descending order
    matches.sort(reverse=True)

    pOut=[]
    gOut=[]
    didpn=[]
    didgn=[]
    # Find all matches with the highest IoU threshold
    for mt in range(len(matches)):
      mbox = matches[mt]
      if mbox[1] not in didpn and mbox[2] not in didgn:
        didpn.append(mbox[1])
        didgn.append(mbox[2])
        pOut.append(mbox[3])
        gOut.append(mbox[4])

    return np.array(pOut), np.array(gOut)
    #END OF YOUR CODE


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    # YOUR CODE HERE
    pred_matches, gt_matches = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    outpt=[]
    outpt.append(pred_matches.shape[0])#true_pos
    outpt.append(prediction_boxes.shape[0] - pred_matches.shape[0])#false_pos
    outpt.append(gt_boxes.shape[0] - pred_matches.shape[0])#false_neg
    #return np.array(outpt)
    return outpt
    #END OF YOUR CODE

def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    # YOUR CODE HERE
    tp=0
    fp=0
    fn=0
    assert(len(all_prediction_boxes) == len(all_gt_boxes))
    for ind in range(len(all_prediction_boxes)):
        pdict=calculate_individual_image_result (all_prediction_boxes[ind], all_gt_boxes[ind], iou_threshold)
        tp+=pdict[0]
        fp+=pdict[1]
        fn+=pdict[2]
    precision=calculate_precision(tp, fp, fn)
    recall=calculate_recall(tp, fp, fn)
    return precision, recall
    #END OF YOUR CODE


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE
    precisions = [] 
    recalls = []
    assert(len(all_prediction_boxes) == len(all_gt_boxes))
    for conf in confidence_thresholds:
      finBoxes = []
      for ind in range(len(all_prediction_boxes)):
        actBox=[]
        for c in range(confidence_scores[ind].shape[0]):
          if confidence_scores[ind][c] >= conf:
            #precision, recall = calculate_precision_recall_all_images(all_prediction_boxes[ind], all_gt_boxes, iou_threshold)
            #precisions.append(precision)
            #recalls.append(recall)
            actBox.append(all_prediction_boxes[ind][c])
        finBoxes.append(np.array(actBox))
      precision, recall = calculate_precision_recall_all_images(finBoxes, all_gt_boxes, iou_threshold)
      precisions.append(precision)
      recalls.append(recall)
    # END OF YOUR CODE

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE
    recall_levels = recall_levels[::-1] #reversing the array, so their elements match with recalls
    average_precision = 0.0
    lu=0 #last used index
    prec=0
    for ind in range(len(recall_levels)):
      top = recall_levels[ind]
      while (lu < len(precisions) and top <= recalls[lu]):
        if precisions[lu] > prec:
          prec=precisions[lu]
        lu += 1
      average_precision += prec
    #END OF YOUR CODE
    return average_precision/len(recall_levels)


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))

def main():
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)

if __name__ == "__main__":
    main()
