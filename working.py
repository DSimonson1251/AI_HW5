import pandas as pd
import os

data_dir = './histopathologic-cancer-detection'

ls = os.listdir(data_dir)
print(ls)

sample = pd.read_csv('histopathologic-cancer-detection/sample_submission.csv')
print(sample.shape)
print(sample.head())

train = pd.read_csv('histopathologic-cancer-detection/train_labels.csv', dtype=str)
print(train.shape)
print(train.head())

# train.id = train.id + '.tif'
# train.head()
import matplotlib.pyplot as plt
import cv2

# Load the labels
labels = pd.read_csv('histopathologic-cancer-detection/train_labels.csv')
print("labels: ", labels.head())

def show_images(images, labels, num=10):
    for i in range(num):
        plt.subplot(2, 5, i+1)
        img_path = os.path.join('histopathologic-cancer-detection/train', images[i] + '.tif')
        img = cv2.imread(img_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Label: {}'.format(labels[i]))
        plt.axis('off')
    plt.show()

show_images(labels['id'].values,labels['label'].values)