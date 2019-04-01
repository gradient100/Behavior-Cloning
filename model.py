import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# steering correction for left camera (right camera is neg value)
correction = 0.2

lines = []
with open('data_merge/driving_log.csv') as csvfile:
	reader  = csv.reader(csvfile)
	for i, line in enumerate(reader):
		if i==0:
			continue
		lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print(lines[0])

def generator(lines, batch_size=32):
	num_samples = len(lines)
	while 1: # Loop forever so the generator never terminates
		shuffle(lines)
		for offset in range(0, num_samples, batch_size):
			batch_samples = lines[offset:offset+batch_size]

			images = []
			measurements = []
			for batch_sample in batch_samples:
				# loop through 3 camera images per point in time
				for j in range(3):
					name = 'data_merge/IMG/' + batch_sample[j].split('/')[-1]
					image = cv2.imread(name)
					image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

					# steering measurement corrections for each camera image
					angle_offset = [0, correction, -correction]
					measurement = float(line[3]) + angle_offset[j]
        
					images.append(image)
					measurements.append(measurement)
                
                	# flip image horizontally for augmented images
					images.append(cv2.flip(image,1))
					measurements.append(measurement*-1.0)
	
			X_train = np.array(images)
			y_train = np.array(measurements)
			
			yield (X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# Trim images to only limit portion of image with road
model.add(Cropping2D(cropping=((70,20), (0,0))))


#Use the Nvidia pipeline, with dropout layers
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.7))
model.add(Dense(50))
model.add(Dropout(0.7))
model.add(Dense(10))
model.add(Dropout(0.7))
model.add(Dense(1))


model.compile(loss = 'mse', optimizer= 'adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)


model.save('model.h5')

