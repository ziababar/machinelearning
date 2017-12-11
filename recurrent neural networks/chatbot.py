

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

# 400 tokens per example, batches of 32. Our word vectors are 300 elements long. Run for 2 epochs again.
maxlen = 400
batch_size = 32
embedding_dims = 300
epochs = 2

# pad/truncate the samples again
import numpy as np

x_train = pad_trunc(x_train, maxlen)
x_test = pad_trunc(x_test, maxlen)

x_train = np.reshape(x_train, (len(x_train), maxlen, embedding_dims))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), maxlen, embedding_dims))
y_test = np.array(y_test)


# build a model. We’ll start as always with a Sequential() Keras model.
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, SimpleRNN

num_neurons = 50

print('Build model...')
model = Sequential()

# sets up the infrastructure to take each input and pass it into a simple RNN (not-simple is in the next chapter) and for each token gather the output into a vector.
model.add(SimpleRNN(num_neurons, return_sequences=True, input_shape=(maxlen, embedding_dims)))

#  to prevent overfitting we add a Dropout layer to zero out 20% of those inputs. Randomly chosen on each input example. And then finally we add a classifier. 
model.add(Dropout(.2))

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
