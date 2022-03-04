from tensorflow.keras.optimizers import Adam
from pandas import read_csv
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x = read_csv('../input/mnist-in-csv/mnist_test.csv')
y = x.label
x.drop(['label'], axis=1, inplace=True)
x /= 255.0

x = x.to_numpy().reshape((x.shape[0], 28, 28, 1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.3)))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])
history = model.fit(x_train, y_train, verbose=1,batch_size=64, epochs=100, validation_split=0.1)

print(model.evaluate(x_test, y_test))
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.plot( history.history['loss'], label='Train loss')
plt.plot( history.history['val_loss'], label = 'Validation loss')
plt.legend()
plt.show()
