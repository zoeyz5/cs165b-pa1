# Starter code for CS 165B HW2 Spring 2019
import numpy as np

def get_w(point1, point2):
    w = []
    for i in range(len(point1)):
        w.append(point1[i] - point2[i])
    return w

def get_t(point1, point2):
    t = 0
    part1 = []
    part2 = []
    for i in range(len(point1)):
        part1.append((point1[i] + point2[i]) / 2)
        part2.append(point1[i] - point2[i])
    t = np.dot(part1, part2)
    return t

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.
    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file
    Output:
        Dictionary of result values
        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED
        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """
    D = training_input[0][0];
    N1 = training_input[0][1];
    N2 = training_input[0][2];
    N3 = training_input[0][3];

    A = training_input[1:N1+1]
    B = training_input[N1+1:N1+N2+1]
    C = training_input[N1+N2+1:-1]

    #training center
    A_center = np.mean(A,axis=0)
    B_center = np.mean(B,axis=0)
    C_center = np.mean(C,axis=0)

    #testing set
    tD = testing_input[0][0];
    tN1 = testing_input[0][1];
    tN2 = testing_input[0][2];
    tN3 = testing_input[0][3];

    tA = testing_input[1:tN1+1]
    tB = testing_input[tN1+1:tN1+tN2+1]
    tC = testing_input[tN1+tN2+1:-1]

    ATP = 0
    AFP = 0
    ATN = 0
    AFN = 0

    BTP = 0
    BFP = 0
    BFN = 0
    BTN = 0

    CTP = 0
    CFN = 0
    CFP = 0
    CTN = 0

    t_AB = get_t(A_center, B_center)
    t_AC = get_t(A_center, C_center)
    t_BC = get_t(B_center, C_center)
    w_AB = get_w(A_center, B_center)
    w_AC = get_w(A_center, C_center)
    w_BC = get_w(B_center, C_center)

    for p in tA:
        if np.dot(w_AB,p) > t_AB:
            if np.dot(w_AC,p) > t_AC:
                ATP += 1
                BTN += 1
                CTN += 1
            else:
                AFN += 1
                BTN += 1
                CFP += 1
        else:
            if np.dot(w_BC,p) > t_BC:
                AFN += 1
                BFP += 1
                CTN += 1
            else:
                AFN += 1
                BTN += 1
                CFP += 1

    for p in tB:
        if np.dot(w_AB, p) > t_AB:
            if np.dot(w_AC, p) > t_AC:
                AFP += 1
                BFN += 1
                CTN += 1
            else:
                ATN += 1
                BFN += 1
                CFP += 1
        else:
            if np.dot(w_BC, p) > t_BC:
                ATN += 1
                BTP += 1

                CTN += 1
            else:
                ATN += 1
                BFN += 1
                CFP += 1

    for p in tC:
        if np.dot(w_AB, p) > t_AB:
            if np.dot(w_AC, p) > t_AC:
                AFP += 1
                BTN += 1
                CFN += 1
            else:
                ATN += 1
                BTN += 1
                CTP += 1
        else:
            if np.dot(w_BC, p) > t_BC:
                ATN += 1
                BFP += 1
                CFN += 1
            else:
                ATN += 1
                BTN += 1
                CTP += 1

    ATPr = ATP/(ATP+AFN)
    BTPr = BTP/(BTP+BFN)
    CTPr = CTP/(CTP+CFN)
    average_tpr = (ATPr+BTPr+CTPr)/3

    AFPr = AFP/(AFP+ATN)
    BFPr = BFP/(BFP+BTN)
    CFPr = CFP/(CFP+CTN)
    average_fpr = (AFPr+BFPr+CFPr)/3

    errA = (AFP+AFN)/(ATP+AFN+AFP+ATN)
    errB = (BFP+BFN) / (BTP+BFN+BFP+BTN)
    errC = (CFP+CFN)/(CTP+CFN+CFP+CTN)
    average_err = (errC+errB+errA)/3

    accA = (ATP+ATN)/(ATP+AFN+AFP+ATN)
    accB = (BTP+BTN) /(BTP+BFN+BFP+BTN)
    accC = (CTP+CTN) /(CTP+CFN+CFP+CTN)
    average_acc = (accC+accA+accB)/3

    perA = ATP/(ATP+AFP)
    perB = BTP/(BTP+BFP)
    perC = CTP/(CTP+CFP)
    average_per = (perC+perB+perA)/3

    d = {}
    d["tpr"] = average_tpr
    d["fpr"] = average_fpr
    d["error_rate"] = average_err
    d["accuracy"] = average_acc
    d["precision"] = average_per
    return d