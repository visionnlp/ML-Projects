# make a prediction on trained model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle


def predict(one_obs, filename):
    # load the model from disk
    model = pickle.load(open(filename, 'rb'))
    y_pred = model.predict(one_obs)
    print(y_pred)