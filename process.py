import pickle
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import GridSearchCV

BOW_FILE_PICKLE = "model/bow_dict.pkl"
SCALER_SIFT_FILE_PICKLE = "model/scaler_sift.pkl"
SVM_SIFT_FILE_PICKLE = "model/svm_model_sift.pkl"

hiragana = ('a','chi','e','fu','ha','he','hi','ho','i','ka','ke','ki','ko','ku','ma','me','mi','mo','mu','n','na','ne','ni','no','nu','o','ra','re','ri','ro','ru','sa','se','shi','so','su','ta','te','to','tsu','u','wa','wo','ya','yo','yu')

# Load Pickle File
def load_file_pickle(filename):
    file_pickle = pickle.load(open(filename, 'rb'))
    return file_pickle

# Load Image File ==================================================================================
def import_image(file) :
    image = cv2.imread(file)
    # image = cv2.bitwise_not(image)
    return image
# ============================================================================================

# Preprocessing Image ========================================================================
def equalizing(img):
    img = cv2.equalizeHist(img)
    return img

def grayscaling(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def resizing(image, size):
    image = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
    return image

def invert_colors(image):
    inverted_image = cv2.bitwise_not(image)
    return inverted_image

def prep_image(image, save_path=None):
    img = resizing(image, 192)
    img = grayscaling(img)
    # Invert colors
    img = invert_colors(img)

    img = equalizing(img)

    # Display the preprocessed image for debugging
    # cv2.imshow("Preprocessed Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the processed image
    if save_path:
        cv2.imwrite(save_path, img)

    return img
# ============================================================================================

# Feature Extraction =========================================================================
def extract_sift_descriptor(image):
    sift = cv2.SIFT_create()
    _, descriptor = sift.detectAndCompute(image, None)
    return descriptor

def create_feature_bow(image_descriptor, bow, num_cluster):
    features = np.array([0] * num_cluster, dtype=float)

    if image_descriptor is not None:
        distance = cdist(image_descriptor, bow)
        argmin = np.argmin(distance, axis = 1)
        
        for j in argmin:
            features[j] += 1.0

    return np.array(features)

def extract_feature(image):
    img_descriptor = extract_sift_descriptor(image)
    
    num_cluster = 1000
    bow = load_file_pickle(BOW_FILE_PICKLE)
    
    img_feature = create_feature_bow(img_descriptor, bow, num_cluster)
    return img_feature
# ============================================================================================

# Prediction Process ======================================================================
def predict_process(filepath):
    # Load file
    img = import_image(filepath)
    
    # Preprocessing Image
    # img = prep_image(img)

     # Preprocessing Image and Save
    processed_image_path = "./static/processed/processed_image.png"  # Define the path where you want to save the image
    img = prep_image(img, save_path=processed_image_path)
    
    # Feature Extraction
    img_feature = extract_feature(img)

    # Feature Scaling
    scaler = load_file_pickle(SCALER_SIFT_FILE_PICKLE)
    feature_scale = scaler.transform([img_feature])
    # print(feature_scale)
    
    # Predict SVM
    svm_model = load_file_pickle(SVM_SIFT_FILE_PICKLE)
    result_predict = svm_model.predict_proba(feature_scale)
    print(result_predict)

    result_label = hiragana[result_predict.argmax()]
    result_accuracy = round(result_predict.max() * 100, 2)
    
    return result_label, result_accuracy
# ============================================================================================