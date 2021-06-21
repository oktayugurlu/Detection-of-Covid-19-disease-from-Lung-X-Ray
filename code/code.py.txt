import cv2
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import math
import operator
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
import matplotlib.pyplot as plt                              #For visualization
import seaborn as sns; sns.set() 

SIZE=64

# Grayscale
def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray

# Canny Edge dedection
def Canny_edge(img):
    canny_edges = cv2.Canny(np.uint8(img),100,200)
    return canny_edges

# Gabor Filter
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get half size
    d = K_size // 2

    # prepare kernel
    gabor = np.zeros((K_size, K_size), dtype=np.float32)

    # each value
    for y in range(K_size):
        for x in range(K_size):
            # distance from center
            px = x - d
            py = y - d

            # degree -> radian
            theta = angle / 180. * np.pi

            # get kernel x
            _x = np.cos(theta) * px + np.sin(theta) * py

            # get kernel y
            _y = -np.sin(theta) * px + np.cos(theta) * py

            # fill kernel
            gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * np.cos(2*np.pi*_x/Lambda + Psi)

    # kernel normalization
    gabor /= np.sum(np.abs(gabor))

    return gabor


# Use Gabor filter to act on the image
def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    gray = np.pad(gray, (K_size//2, K_size//2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)
        
    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y : y + K_size, x : x + K_size] * gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


# Use 6 Gabor filters with different angles to perform feature extraction on the image
def Gabor_process(img):
    # get shape
    H, W, _ = img.shape

    # gray scale
    gray = BGR2GRAY(img).astype(np.float32)

    # define angle
    #As = [0, 45, 90, 135]
    As = [0,30,60,90,120,150]

    # prepare pyplot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = Gabor_filtering(gray, K_size=9, Sigma=1.5, Gamma=1.2, Lambda=1, angle=A)

        # add gabor filtered image
        out += _out

    # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out


#LOAD IMAGES, Please give the images to read
image_list_COVID = []
out_list_COVID_not_filtered = np.empty((0,SIZE*SIZE), dtype=float, order='C')
for i in range(1,961):
    # Read image
    imgCOVID=cv2.imread('dataset/COVID/COVID ('+str(i)+').PNG').astype(np.float32)
    image_list_COVID.append(imgCOVID)
    
image_list_NORMAL = []
for i in range(1,1074):
    # Read image
    img=cv2.imread('dataset/NORMAL/NORMAL ('+str(i)+').PNG').astype(np.float32)
    image_list_NORMAL.append(img)
    
image_list_VIRAL = []
for i in range(1,1077):
    # Read image
    imgVIRAL=cv2.imread('dataset/Viral Pneumonia/Viral Pneumonia ('+str(i)+').png').astype(np.float32)
    image_list_VIRAL.append(imgVIRAL)
    
    
    
    
canny_out_list_COVID = np.empty((0,SIZE*SIZE), dtype=float, order='C')
for i in range(0,960):
    #canny edge process
    out_canny = Canny_edge(cv2.resize(image_list_COVID[i], (SIZE, SIZE)))
    canny_out_list_COVID = np.append(canny_out_list_COVID, out_canny.reshape(1, SIZE*SIZE), axis=0)

canny_out_list_NORMAL = np.empty((0,SIZE*SIZE), dtype=float, order='C')
for i in range(0,1073):
    #canny edge process
    out_canny = Canny_edge(cv2.resize(image_list_NORMAL[i], (SIZE, SIZE)))
    canny_out_list_NORMAL = np.append(canny_out_list_NORMAL, out_canny.reshape(1, SIZE*SIZE), axis=0)

canny_out_list_VIRAL = np.empty((0,SIZE*SIZE), dtype=float, order='C')
for i in range(0,1076):
    #canny edge process
    out_canny = Canny_edge(cv2.resize(image_list_VIRAL[i], (SIZE, SIZE)))
    canny_out_list_VIRAL = np.append(canny_out_list_VIRAL, out_canny.reshape(1, SIZE*SIZE), axis=0)
    
    
#gabor featured data
out_list_COVID = pd.read_csv ('COVID.csv', engine='python')
out_list_VIRAL = pd.read_csv ('VIRAL.csv', engine='python')
out_list_NORMAL = pd.read_csv ('NORMAL.csv', engine='python')

label_COVID_df = np.empty((960,1), dtype=float, order='C')
label_COVID_df[:] = 1

label_NORMAL_df = np.empty((1073,1), dtype=float, order='C')
label_NORMAL_df[:] = 2

label_VIRAL_df = np.empty((1076,1), dtype=float, order='C')
label_VIRAL_df[:] = 3

composed_feature_arr = np.concatenate((out_list_COVID, out_list_NORMAL))
composed_feature_arr = np.concatenate((composed_feature_arr, out_list_VIRAL))

composed_label_arr = np.concatenate((label_COVID_df, label_NORMAL_df))
composed_label_arr = np.concatenate((composed_label_arr, label_VIRAL_df))
composed_feature_arr = pd.DataFrame(composed_feature_arr)
composed_feature_arr[4096] = composed_label_arr
gabor_composed_data = composed_feature_arr
not_filtered_composed_feature_arr = np.concatenate((out_list_COVID_not_filtered, out_list_NORMAL_not_filtered))
not_filtered_composed_feature_arr = np.concatenate((not_filtered_composed_feature_arr, out_list_VIRAL_not_filtered))

not_filtered_composed_feature_arr = pd.DataFrame(not_filtered_composed_feature_arr)
not_filtered_composed_feature_arr[4096] = composed_label_arr
not_filtered_composed_data = not_filtered_composed_feature_arr

not_filtered_gabor_composed_data = pd.concat([not_filtered_composed_data,gabor_composed_data], axis = 1)

canny_composed_feature_arr = np.concatenate((canny_out_list_COVID, canny_out_list_NORMAL))
canny_composed_feature_arr = np.concatenate((canny_composed_feature_arr, canny_out_list_VIRAL))

canny_composed_feature_arr = pd.DataFrame(canny_composed_feature_arr)
canny_composed_feature_arr[4096] = composed_label_arr
canny_composed_data = composed_feature_arr

canny_gabor_composed_data = pd.concat([canny_composed_data,gabor_composed_data], axis = 1)

not_filtered_canny_gabor_composed_data = pd.concat([canny_gabor_composed_data,not_filtered_composed_data], axis = 1)

class KNN_Classification:
    trained_data = []
    #prediction_and_row_list is used to get index of tested data and its predicted label, after using it in error analysis 
    
    def __init__(self, k, prediction_and_row_index_list=[]):
        self.k = k
        self.prediction_and_row_index_list = []
    
    # Calculating distance
    def calculate_distances(self, trained_data_list, test_data):
        distance_array = np.sqrt(np.sum(np.square(np.subtract(trained_data_list, test_data)), axis = 1))
        return distance_array.reshape(distance_array.shape[0],-1)

    def check_key_existance_and_return_value(self, dictionary,key):
        if key in dictionary:
            return dictionary[key]
        else:
            return 0

    # Because of performance issues, I convert dataframes to numpy arrays to increase the calculation of distances.
    def dataframe_to_numpy(self, trained_data, trained_label, test_data):
        return (trained_data.to_numpy(), trained_label.to_numpy(), test_data.to_numpy())

    # Train
    def train(self, trained_data, trained_label):
        self.trained_data = trained_data.copy()
        self.trained_label = trained_label.copy()
        self.index_array = np.array([i for i in range(trained_data.shape[0])])
        #I convert 1-D data to 2-D data
        self.index_array = self.index_array.reshape(self.index_array.shape[0],-1)
    
    # Knn predict. test data should not contain label column
    def predict(self, test_data):
        prediction_list = []
        trained_data_array, trained_label_array, test_data_array = self.dataframe_to_numpy(self.trained_data, self.trained_label, test_data)
        for test_data_i in range(test_data_array.shape[0]):
            distance_index_array = self.calculate_distances(trained_data_array, test_data_array[test_data_i,:])
            # index_array which keeps indexes of used train data used in distance calculation are 
            # added as second column to find the label of this point easily in trained_label_array
            distance_index_array = np.append(distance_index_array, self.index_array, axis=1)
            
            nearest_n_distance_index_array = distance_index_array[distance_index_array[:,0].argsort()]
            label_numbers_dict = {}
            
            for nearest_n_distance_array_i in range(self.k):
                label_of_nearest_data = trained_label_array[int(nearest_n_distance_index_array[nearest_n_distance_array_i,1])]
                label_numbers_dict[label_of_nearest_data] = self.check_key_existance_and_return_value(label_numbers_dict, label_of_nearest_data) + 1
            predicted_label = max(label_numbers_dict.items(), key=operator.itemgetter(1))[0]
            prediction_list.append(predicted_label)
            self.prediction_and_row_index_list.append([test_data.index[test_data_i], predicted_label])
        return pd.DataFrame(prediction_list)
    
    
class Weighted_KNN_Classification:
    trained_data = []
    #prediction_and_row_list is used to get index of tested data and its predicted label, after using it in error analysis 
    
    def __init__(self, k, prediction_and_row_index_list=[]):
        self.k = k
        self.prediction_and_row_index_list = []
    
    # Train
    def train(self, trained_data, trained_label):
        self.trained_data = trained_data.copy()
        self.trained_label = trained_label.copy()
        self.index_array = np.array([i for i in range(trained_data.shape[0])])
        self.index_array = self.index_array.reshape(self.index_array.shape[0],-1)
    
    # Calculating distance
    def calculate_distances(self, trained_data_list, test_data):
        distance_array = np.sqrt(np.sum(np.square(np.subtract(trained_data_list, test_data)), axis = 1))
        return distance_array.reshape(distance_array.shape[0],-1)
    
   # inverse square distance formula
    def calculate_weight(self, distance):
        return 1 / (distance)
    
    def check_key_existance_and_return_value(self, dictionary,key):
        if key in dictionary:
            return dictionary[key]
        else:
            return 0
        
    def sum_weights_of_each_group(self, distance_and_index_array, label_array):
        label_weight_dict = {}
        for row in range(self.k):
            label = label_array[int(distance_and_index_array[row,1])]
            label_weight_dict[label] = self.check_key_existance_and_return_value(label_weight_dict, label) + self.calculate_weight(distance_and_index_array[row,0])
        return label_weight_dict

    def convert_dataframe_to_numpy(self, trained_data, trained_label, test_data):
        return (trained_data.to_numpy(), trained_label.to_numpy(), test_data.to_numpy())
    
    # Knn predict. test data should not contain label column
    def predict(self, test_data):
        prediction_list = []
        trained_data_array, trained_label_array, test_data_array = self.convert_dataframe_to_numpy(self.trained_data, self.trained_label, test_data)
        for test_data_i in range(test_data_array.shape[0]):
            distance_index_array = self.calculate_distances(trained_data_array, test_data_array[test_data_i,:])

            # Indexes of used train data used in distance calculation are added as second column
            # to find the label of this point easily in trained_label_array
            distance_index_array = np.append(distance_index_array, self.index_array, axis=1)
            nearest_n_distance_index_array = distance_index_array[distance_index_array[:,0].argsort()][:self.k]
            
            label_numbers_dict = self.sum_weights_of_each_group(nearest_n_distance_index_array, trained_label_array)
            predicted_label = max(label_numbers_dict.items(), key=operator.itemgetter(1))[0]
            prediction_list.append(predicted_label)
            self.prediction_and_row_index_list.append([test_data.index[test_data_i], predicted_label])
        return pd.DataFrame(prediction_list)       
    
# I calculate the accuracy by dividing the number of correctly classified examples of number of examples

def accuracy(predicted_labels, actual_label):
    # Because the actual labels index is not start from 0, I resett its index
    actual_label = actual_label.reset_index(drop=True).to_frame()
    actual_label.columns = [0]
    difference = predicted_labels.sub(actual_label)
    return difference[difference.iloc[:,0]==0].shape[0] / actual_label.shape[0]

def split_data(data_array, k):
    data_number_in_each_bucket = len(data_array)//k
    start_index_of_bucket=0
    end_index_of_bucket=data_number_in_each_bucket
    splitted_data_list=[]
    for i in range(0,k):
        if(i==k-1):
            splitted_data_list.append(data_array[start_index_of_bucket:])
        else:
            splitted_data_list.append(data_array[start_index_of_bucket:end_index_of_bucket])
            start_index_of_bucket = end_index_of_bucket
            end_index_of_bucket += data_number_in_each_bucket
    return splitted_data_list

def k_fold_cross_validation(model, data_df, k=5):
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    splitted_data_list = split_data(data_df, k)
    total_accuracy = 0
    for test_data_index in range(0,k):
        train_data_list = splitted_data_list.copy()
        train_data_list.pop(test_data_index)
        
        train_data = pd.concat(train_data_list, ignore_index=True)
        test_data = splitted_data_list[test_data_index]
        knn.train(train_data.iloc[:,:-1], train_data.iloc[:,-1])
        prediction_array = knn.predict(test_data.iloc[:,:-1])
        total_accuracy += accuracy(prediction_array, test_data.iloc[:,-1])
    return total_accuracy/k

gabor_knn = KNN_Classification(11)
gabor_knn_accuracy = k_fold_cross_validation(gabor_knn, gabor_composed_data)
print("K-Fold Accuracy: "+ str(gabor_knn_accuracy))

weighted_gabor_knn = Weighted_KNN_Classification(5)
weighted_gabor_knn_accuracy = k_fold_cross_validation(weighted_gabor_knn, gabor_composed_data)
print("K-Fold Accuracy: "+ str(weighted_gabor_knn_accuracy))

not_filtered_gabor_knn = KNN_Classification(11)
not_filtered_gabor_knn_accuracy=k_fold_cross_validation(not_filtered_gabor_knn, not_filtered_gabor_composed_data)
print("K-Fold Accuracy: "+ str(not_filtered_gabor_knn_accuracy))

weighted_not_filtered_gabor_knn = Weighted_KNN_Classification(7)
weighted_not_filtered_gabor_knn_accuracy=k_fold_cross_validation(weighted_not_filtered_gabor_knn, not_filtered_gabor_composed_data)
print("K-Fold Accuracy: "+ str(weighted_not_filtered_gabor_knn_accuracy))

not_filtered_canny_gabor_knn = KNN_Classification(5)
not_filtered_canny_gabor_knn_accuracy= k_fold_cross_validation(not_filtered_canny_gabor_knn, not_filtered_canny_gabor_composed_data)
print("K-Fold Accuracy: "+ str(not_filtered_canny_gabor_knn_accuracy))

not_filtered_canny_gabor_knn = Weighted_KNN_Classification(17)
not_filtered_canny_gabor_knn_accuracy= k_fold_cross_validation(not_filtered_canny_gabor_knn, not_filtered_canny_gabor_composed_data)
print("K-Fold Accuracy: "+ str(not_filtered_canny_gabor_knn_accuracy))
