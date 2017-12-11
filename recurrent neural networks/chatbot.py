

# Load the dataset and grab the labels and shuffle the examples.
# Tokenize it and vectorize it again using the Google word2vec model.
# Grab the labels off in an ordered set.
dataset = pre_process_data('./aclimdb/train')
vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)

# Split it 80/20 into the training and test sets.
# Divide up the train and test sets
split_point = int(len(vectorized_data)*.8)

x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]

