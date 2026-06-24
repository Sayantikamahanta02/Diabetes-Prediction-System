import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('diabetes.csv')
print(df)
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df.dropna)
x=df.drop(['Outcome'],axis=1)
y=df[['Outcome']]
# print(x.shape)  
# print(y.shape)
x_test,x_train,y_test,y_train=train_test_split(
    x,y,test_size=0.2,random_state=42
    )
model1=LogisticRegression()
model1.fit(x_train,y_train)
predict_model1=model1.predict(x_test)
# # print(predict_result)
acc_score1=accuracy_score(y_test,predict_model1)
# recall1=recall_score()
acc_recall1=recall_score(y_test,predict_model1) 
acc_presion1=precision_score(y_test,predict_model1) 
categories =['Accuracy_score','Recall_score','Precision_score'] 
value=[acc_score1,acc_recall1,acc_presion1] 
plt.bar(categories,value,color='skyblue') 
plt.xlabel('Score') 
plt.title('Accuracy Score') 
plt.show()
print("Logistic Regression Accuracy:", acc_score1)  
print("Logistic Regression Recall Accuracy:", acc_recall1)  
print("Logistic Regression Presion Accuracy:", acc_presion1)  
model2=DecisionTreeClassifier()
model2.fit(x_train,y_train)
predict_model2=model2.predict(x_test)
acc_score2=accuracy_score(y_test,predict_model2)
acc_recall2=recall_score(y_test,predict_model2) 
acc_presion2=precision_score(y_test,predict_model2) 
categories =['Accuracy_score','Recall_score','Precision_score'] 
value=[acc_score2,acc_recall2,acc_presion2] 
plt.bar(categories,value,color='skyblue') 
plt.xlabel('Score') 
plt.title('Accuracy Score') 
plt.show()
print("Decision Tree Accuracy:",acc_score2)
print("Decision Tree Recall Accuracy:", acc_recall2)  
print("Logistic Regression Accuracy:", acc_presion2)  
model3= RandomForestClassifier()
model3.fit(x_train,y_train)
predict_model3=model3.predict(x_test)
acc_score3=accuracy_score(y_test,predict_model3)
acc_recall3=recall_score(y_test,predict_model3) 
acc_presion3=precision_score(y_test,predict_model3) 
categories =['Accuracy_score','Recall_score','Precision_score'] 
value=[acc_score3,acc_recall3,acc_presion3] 
plt.bar(categories,value,color='skyblue') 
plt.xlabel('Score') 
plt.title('Accuracy Score') 
plt.show()
print("Randome Forest Accuracy:",acc_score3)
print("Randome ForestRecall Accuracy:", acc_recall3)  
print("Randome Forest Accuracy:", acc_presion3)  
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracy = [acc_score1,acc_score2,acc_score3]
plt.bar(models, accuracy)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()
Pregnancies=int(input("Enter the Number of pregnancies:"))
Glucose=int(input("Enter the Number of Blood sugar level:"))
BloodPressure=int(input("Enter the Number of Blood pressure:"))
Skinthickness=int(input("Enter the Number of Skin thickness:"))
Insulin=int(input("Enter the Number of Insulin level:"))
BMI=float(input("Enter the Number of Body Mass Index:"))
DiabetesPedigreeFunction=float(input("Enter the Number of Family diabetes history:"))
Age=int(input("Enter the Number of Person age:"))
user=[[Pregnancies,Glucose,BloodPressure,Skinthickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
result=model1.predict(user)
if result==1:
    print("Diabetes")
else:
    print("No Diabetes")  