import math
import sys

def initialLabels(input):
    contents = {}
    with open(input, 'r') as inputlabels:
        contents = [k.strip('\n') for k in inputlabels.readlines()]
    y_vector = []
    for datapoint in contents:
        data = datapoint.split('\t')
        label = data[0]
        y_vector.append(int(label))
    return y_vector

def reformatdata(input):
    contents = {}
    with open(input, 'r') as inputlabels:
        contents = [k.strip('\n') for k in inputlabels.readlines()]
    output = {}
    output_data = []
    pair = ""
    key = 0
    value = 0
    for datapoint in contents:
        data = datapoint.split('\t')
        for val in data[1:]:
            pair = val.split(":")
            key = int(pair[0])
            value = int(pair[1])
            output.update({key:value})
        output_data.append(output)
        output = {}
    return output_data

def sig(val):
    return math.exp(val)/(math.exp(val) + 1.0)

def zerovector(dict_length):
    initdata = []
    for i in range(dict_length):
        initdata.append(0)
    return initdata

def inputXmatrix(dict_length,output_data):
    Xmatrix = []
    inputvector = []
    for datapoint in output_data:
        inputvector = zerovector(dict_length+1)
        for k,v in datapoint.items():
            inputvector[k] = 1
        inputvector[-1] = 1
        Xmatrix.append(inputvector)
    return Xmatrix

def sparse_dot(x, w):
    product = 0.0
    for i, v in x.items():
        product += w[i] * v
    return product

def sparse_dot1(x, weights,delta,learning_rate):
    for i, v in x.items():
        weights[i] += float(v)*delta*learning_rate
    return weights

def sgd(output_data,Xmatrix, labels,epochs, dict_length,learning_rate = 0.1):
    weights = zerovector(dict_length+1)
    for e in range(epochs):
        for i in range(len(Xmatrix)):
            dot_product = sparse_dot(output_data[i],weights)
            inner = float(labels[i]) - sig(dot_product)
            weights = sparse_dot1(output_data[i],weights,inner,learning_rate)
            weights[-1] += learning_rate*inner
    return weights

def predict(output_data, weights, labels1):
    bias = weights[-1]
    labels = []
    for dk in output_data:
        labels.append(int(round(sig(sparse_dot(dk, weights)+bias))))
    count = 0
    error = 0
    for l in range(len(labels)):
        if(labels[l] != labels1[l]):
            count += 1
    error = count/(len(labels))
    results = map(str, labels)
    finalresult = "\n".join(results)
    finalresult = finalresult + "\n"
    return labels, finalresult, error

def neg_log_likelihood(output_data, labels, weights):
    neg_log = 0.0
    bias = weights[-1]
    data_length = len(output_data)
    for i in range(data_length):
        sig_val = sparse_dot(output_data[i], weights)+bias
        neg_log += (math.log(1.0 + math.exp(sig_val)) + (-(labels[i]) * sig_val))
    return neg_log / data_length


def sgdwrittenvalid(output_data,Xmatrix, labels,valid_data,valid_labels,epochs, dict_length,learning_rate = 0.1):
    neg_log_valid = []
    weights = zerovector(dict_length+1)
    for e in range(epochs):
        for i in range(len(Xmatrix)):
            dot_product = sparse_dot(output_data[i],weights)
            inner = labels[i] - sig(dot_product)
            weights = sparse_dot1(output_data[i],weights,inner,learning_rate)
            weights[-1] += learning_rate*inner
        neg_log_valid.append(neg_log_likelihood(valid_data,valid_labels,weights))
    return neg_log_valid


if __name__ == "__main__":

    contents = []
    with open(str(sys.argv[4]), 'r') as dict_file:
        contents = dict_file.readlines()
    dict_length = len(contents)
    weights = zerovector(dict_length+1)

    train_labels = initialLabels(sys.argv[1])
    train_data = reformatdata(sys.argv[1])
    
    valid_labels = initialLabels(sys.argv[2])
    valid_data = reformatdata(sys.argv[2])
    
    test_labels = initialLabels(sys.argv[3])
    test_data = reformatdata(sys.argv[3])

    Xmatrix_train = inputXmatrix(dict_length, train_data)

    #w,t,v = sgdwritten(train_data,Xmatrix_train, train_labels,valid_data,valid_labels, int(sys.argv[8]), dict_length)

    weight_vector = sgd(train_data,Xmatrix_train, train_labels, int(sys.argv[8]), dict_length)

    new_train_labels, train_output, trainError = predict(train_data, weight_vector, train_labels)
    new_test_labels, test_output, testError = predict(test_data, weight_vector, test_labels)

    # write outputs
    with open(sys.argv[5], 'w') as trainfile:
        trainfile.write(train_output)

    with open(sys.argv[6], 'w') as testfile:
        testfile.write(test_output)

    with open(sys.argv[7], 'w') as errorfile:
        errorfile.write('error(train): ' + str(trainError) + '\n')
        errorfile.write('error(test): ' + str(testError))


