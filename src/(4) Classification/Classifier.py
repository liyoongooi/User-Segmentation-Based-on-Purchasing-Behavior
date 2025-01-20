from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class Classifier:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.X_train = None
        self.classifier = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.overfit = None
        
        self.X_train, X_test, y_train, y_test = train_test_split(data.drop(['Cluster'], axis = 1), data['Cluster'], test_size=0.3, random_state=0)
        
        if self.name == 'Logistic Regression':
            self.classifier = LogisticRegression(random_state = 0)
        elif self.name == 'Random Forest':
            self.classifier = RandomForestClassifier(random_state = 0)
        elif self.name == 'Light Gradient-Boosting Machine':
            self.classifier = LGBMClassifier(random_state = 0)
            
        self.classifier.fit(self.X_train, y_train)
        
        predictions_train = self.classifier.predict(self.X_train)     
        accuracy_train = accuracy_score(y_train, predictions_train)
        
        predictions_test = self.classifier.predict(X_test)
        self.accuracy = accuracy_score(y_test, predictions_test)
        self.precision, self.recall, self.f1, _ = precision_recall_fscore_support(y_test, predictions_test, average='weighted')
        
        if abs(accuracy_train - self.accuracy) <= 0.05:
            self.overfit = False
        else:
            self.overfit = True