import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

#Load dataset
data = pd.read_csv("dataset/music_features.csv")

#split features and labels
X = data.drop("genre",axis=1)
Y = data["genre"]

#split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#create model
model = RandomForestClassifier()

#train model
model.fit(X_train, Y_train)

#predict 
predictions = model.predict(X_test)

#accuracy 
accuracy = accuracy_score(Y_test, predictions)

print("Model Accuracy:", accuracy)

#save model
joblib.dump(model,"models/music_genre_model.pkl")

joblib.dump(model,"models/model.pkl")

print("Model saved successfully!")