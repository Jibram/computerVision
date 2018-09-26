from PIL import Image
import os
from glob import glob
import numpy as np
from skimage import io
import time


# 92x112 but for the sake of numpy array, 112 by 92 (vertical by horizontal)
# rows by columns
# so 10000, 0 is 10000 rows 1 column
# I want to add more columns to the side of the matrix


def list_to_matrix(image_list):
    image_matrix = np.empty((image_list[0].shape[0],len(image_list)))
    for i in range(len(image_list)):
        for j in range(image_list[i].shape[0]):
            image_matrix[j][i] = image_list[i][j]
    return image_matrix


def train_images_to_list():
    image_list = []
    for filename in glob('../train/*.tif'):
        im = Image.open(filename)
        array = np.array(im)
        array = np.reshape(array, len(array) * len(array[0]))
        image_list.append(array)
    image_matrix = list_to_matrix(image_list)
    return image_matrix


def test_images_to_list():
    image_list = []
    for filename in glob('../test/*.tif'):
        im = Image.open(filename)
        array = np.array(im)
        array = np.reshape(array, len(array) * len(array[0]))
        image_list.append(array)
    image_matrix = list_to_matrix(image_list)
    return image_matrix


def center_data(data):
    average = data.sum(axis = 1)/data.shape[1]
    average.reshape((average.shape[0], 1))
    return data


def at_a_trick(data):
    return np.dot(data.transpose(),data)


def eigenspace(eigvals,eigvecs):
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:,idx]
    return eigvecs[:,0:10].real


def train_trick():
    # Step 1
    data = train_images_to_list()
    # Step 2
    datac = center_data(data)
    # Step 3 PCA sucks, do ATA
    ata = at_a_trick(datac) # this is a 40x40
    # Step 4
    eigvals, eigvecs = np.linalg.eig(ata)
    # Step 5
    ataeigvecs = eigenspace(eigvals, eigvecs) # v1
    projmatrix = np.dot(data, ataeigvecs) # A dot v1 = cov eig vecs
    # Step 6
    xp = np.dot(projmatrix.transpose(), datac)
    return projmatrix, xp, data


def train_svd():
    # Step 1
    data = train_images_to_list()
    # Step 2
    datac = center_data(data)
    # Step 3 PCA sucks, do SVD
    U, S, V = np.linalg.svd(datac)
    # step 4
    projmatrix = np.dot(data, V) # A dot v1 = cov eig vecs
    # Step 5
    xp = np.dot(projmatrix.transpose(), datac)
    return projmatrix, xp, data



def test(projmatrix, projected_train, traindata):
    testdata = test_images_to_list()
    testdatac = center_data(testdata)
    projected_test = np.dot(projmatrix.transpose(), testdatac)
    projected_test = projected_test.transpose()
    projected_train = projected_train.transpose()
    dist_matrix = np.empty((projected_test.shape[0], projected_train.shape[0]))
    matches = np.empty((projected_test.shape[0], 3))
    for i in range(projected_test.shape[0]):
        for j in range(projected_train.shape[0]):
            diff_array = projected_train[j] - projected_test[i]
            dist_matrix[i][j] = np.linalg.norm(diff_array, axis = 0)
    for i in range(projected_test.shape[0]):
        matches[i] = dist_matrix[i].argsort()[:3]

    # save three matches
    testdata = testdata.transpose()
    traindata = traindata.transpose()
    for i in range(matches.shape[0]):
        testimage = testdata[i].reshape([112, 92])
        testimage = testimage.astype(int)
        io.imsave('./result_trick/' + str(i) + '_0.jpg', testimage)
        matches_image = traindata[int(matches[i][0])].reshape([112, 92])
        for j in range(1, 3):
            image_as_array = traindata[int(matches[i][j])].reshape([112, 92])
            matches_image = np.hstack((matches_image, image_as_array))
            matches_image = matches_image.astype(int)
        io.imsave('./result_trick/' + str(i) + '_1.jpg', matches_image)


def pca_trick():
    projmatrix, xp, traindata = train_trick()
    test(projmatrix, xp, traindata)


def pca_svd():
    projmatrix, xp, traindata = train_svd()
    test(projmatrix, xp, traindata)

start_time = time.time()
pca_trick()
print("--- %s seconds ---" % (time.time() - start_time))