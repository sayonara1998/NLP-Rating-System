import sys

'''
Input: dict.txt
Output: output dict:
            keys = word
            value = index within dict.txt
'''
def formatDict(dictfile):
    contents = {}
    output = {}
    x = ''
    with open(dictfile) as d:
        contents = [k.strip('\n') for k in d.readlines()]
        for k in contents:
            x = k.split(' ')
            output.update({x[0]: x[1]})
    return output

'''
Input: (training data, result file, formatted dictionary, threshold)
Output: formatted data file according to Model 1/2
'''
def parseData(input,output,dict,trimvalue):
    model = ""
    words = []
    wordsdict = {}
    wordsdictcopy = {}
    contents = []
    with open(input,'r') as infile:
        contents = infile.readlines()
    for k in contents:
        label,features = k.split('\t')
        features.strip('\n')
        words = features.split()
        for w in words:
            if(w in dict and dict[w] in wordsdict):
                wordsdict[dict[w]] += 1
            elif(w in dict):
                wordsdict.update({dict[w] : 1})
        for k,v in wordsdict.items():
            if(trimvalue == 0):
                wordsdictcopy.update({k:1})
            else:
                if(v < trimvalue):
                    wordsdictcopy.update({k:1})
        model += label
        for k,v in wordsdictcopy.items():
            model += "\t" + str(k) + ":" + str(v)
        model += '\n'
        wordsdict = {}
        wordsdictcopy = {}

    with open(output, 'w') as out:
        out.write(model)

if __name__ == "__main__":
    train_input = sys.argv[1]
    vaildation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input  = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = int(sys.argv[8])

    dict_formatted = formatDict(dict_input)

    if feature_flag == 1:
        parseData(train_input,formatted_train_out,dict_formatted,0)
        parseData(vaildation_input, formatted_validation_out, dict_formatted, 0)
        parseData(test_input, formatted_test_out, dict_formatted, 0)
    if feature_flag == 2:
        parseData(train_input, formatted_train_out, dict_formatted, 4)
        parseData(vaildation_input, formatted_validation_out, dict_formatted, 4)
        parseData(test_input, formatted_test_out, dict_formatted, 4)
