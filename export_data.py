import numpy as np

fileDataPath  = 'imageExport.npy'
fileLabelPath = 'labelExport.npy'

dataRead = np.load(fileDataPath)
labelRead = np.load(fileLabelPath)
unique_set = set(labelRead)
xMax = np.max(dataRead)
dataRead = dataRead/xMax
xMax = np.max(dataRead)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dataRead, labelRead, test_size=0.2, shuffle=True, random_state=42)

np.savetxt("hls/X.txt", x_test[:50].reshape(-1), fmt="%.6f")
np.savetxt("hls/Y.txt", y_test[:50], fmt="%.6f")

import matplotlib.pyplot as plt
i = 47
print(f"Shape of y_test[{i}]:", y_test[i].shape)
print(f"Type of y_test[{i}]:", type(y_test[i]))
print(f"Value of y_test[{i}]:", y_test[i])


if y_test[i].ndim == 2:
    plt.imshow(y_test[i], cmap='gray')  
elif y_test[i].ndim == 1:
    shape = x_test[i].shape
    plt.imshow(y_test[i].reshape(shape), cmap='gray')
else:
    plt.imshow(x_test[i], cmap='gray')
plt.axis('off')  
plt.savefig('first_y_test_image.png', bbox_inches='tight', pad_inches=0)
plt.close()  