import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import string
import seaborn as sns

data=pd.read_csv("train_kaggle.csv")

# i want to see the list of features and description the data:
print (data.describe (include="all"))
print (data.columns.values)

# check the missing data in our database :
print (data.isnull().sum().sort_values(ascending=False))

# the number of Male and Female :
print (data["Sex"].value_counts())
print (data["Sex"].value_counts(normalize=True)*100) # the rate of Male and Female in all Passengers.


# Analysis the data:
#Sex Feature:
print ("the Percentage of Gender who are died :\n",data["Sex"][data["Survived"]==0].value_counts(normalize=True)*100)
print ("the Percentage of Gender who are Survived :\n",data["Sex"][data["Survived"]==1].value_counts(normalize=True)*100)
print ("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n Pclass Feature:\n")
print ("The percentage of Gender by Pclass 1 :\n ",data["Sex"][data["Pclass"]==1].value_counts(normalize=True)*100)
print ("The percentage of Gender by Pclass 2 : \n",data["Sex"][data["Pclass"]==2].value_counts(normalize=True)*100)
print ("The percentage of Gender by Pclass 3 : \n",data["Sex"][data["Pclass"]==3].value_counts(normalize=True)*100)
print ("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n SibSp Feature:\n")
for i in sorted(data.SibSp.unique()):
    print ("The Percentage of Gender with number of Sibling and Spouse =",i)
    print (data["Sex"][data["SibSp"]==i].value_counts(normalize=True)*100)
    
print ("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n Parch Feature:\n")    
for i in sorted (data.Parch.unique()):
    print ("The Percentage of Gender with number of Parent and Child=",i)
    print (data["Sex"][data["Parch"]==i].value_counts(normalize=True)*100)
print ("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n Age Feature:\n")
# fill in missing data in Age Group Feature : 
mean = data["Age"].mean()
std = data["Age"].std()
is_null = data["Age"].isnull().sum()

np.random.seed(42)
# compute random numbers between the mean, std and is_null
range_age=np.random.randint(mean - std, mean + std, size = is_null)
data=data.fillna({"Age":np.random.randint(np.min(range_age),np.max(range_age))})

category=pd.cut(data["Age"],[0,18,80],labels=["child(0-18)","Adult(18-80)"])    
data.insert(5,"Age Group",category)
print ("The percentage of Gender in Age between 0- 18 : \n",data["Sex"][data["Age Group"]=="child(0-18)"].value_counts(normalize=True)*100)
print ("The percentage of Gender in Age between 18-80 : \n",data["Sex"][data["Age Group"]=="Adult(18-80)"].value_counts(normalize=True)*100)
#res=data.groupby(["Age Group"])["Sex"].count()# To know all the child and adults passengers .

print ("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n Embarked Feature:\n")
print ("Number of all passengers embarking in Southampton(S):",data[data["Embarked"]=="S"].shape[0]) # 644
print ("Number of all passengers embarking in Cherbourg(C):",data[data["Embarked"]=="C"].shape[0])  # 168
print ("Number of all passengers embarking in Queenstown(Q):",data[data["Embarked"]=="Q"].shape[0]) #77
data=data.fillna({"Embarked":"S"})

print ("The percentage of Gender embarking in Southampton :\n ",data["Sex"][data["Embarked"]=="S"].value_counts(normalize=True)*100)
print ("The percentage of Gender embarking in Cherbourg :\n ",data["Sex"][data["Embarked"]=="C"].value_counts(normalize=True)*100)
print ("The percentage of Gender embarking in Queenstown :\n ",data["Sex"][data["Embarked"]=="Q"].value_counts(normalize=True)*100)
print ("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n Fare Feature:\n")
fare=pd.cut(data["Fare"],[-1,100,200,300,400,513],labels=["0-100","100-200","200-300","300-400","400-513"])    
data.insert(5,"Fare Group",fare)
print ("The percentage of Gender by the price of ticket 0-100 $ :\n ",data["Sex"][data["Fare Group"]=="0-100"].value_counts(normalize=True)*100)
print ("The percentage of Gender by the price of ticket 100-200 $ :\n ",data["Sex"][data["Fare Group"]=="100-200"].value_counts(normalize=True)*100)
print ("The percentage of Gender by the price of ticket 200-300 $ :\n ",data["Sex"][data["Fare Group"]=="200-300"].value_counts(normalize=True)*100)
print ("The percentage of Gender by the price of ticket 300-400 $ :\n ",data["Sex"][data["Fare Group"]=="300-400"].value_counts(normalize=True)*100)
print ("The percentage of Gender by the price of ticket 400-513 $ :\n ",data["Sex"][data["Fare Group"]=="400-513"].value_counts(normalize=True)*100)

#Visualize and analysis the data :
plt.figure()
features=["Survived","Pclass","SibSp","Parch","Age Group","Embarked"]
rows=2
cols=3
fig, axes = plt.subplots(rows, cols, figsize=(cols*4,rows*4))
for i in range(0,rows):
    for j in range(0,cols):
        r=i*cols+j
        ax=axes[i][j] 
        sns.countplot(x=data[features[r]],data=data,hue=data.Sex,ax=ax)
fig.savefig("plot.png") # to save the figure .
plt.tight_layout()
plt.figure()
sns.countplot(x="Fare Group",data=data,hue="Sex")
plt.legend(title="Sex",loc="upper right")
plt.savefig("Fare.png")

data=data.drop(["PassengerId","Cabin","Ticket","Age Group","Fare Group"],axis=1)

# remove the punctuation:  
x=[]
for i in data.Name:
    word=i.translate(str.maketrans("","",string.punctuation))
    word=word.lower()
    x.append(word)
data["Name"]=x
# convert the male and female to numerical values: 
data.Sex.replace({"female":0,"male":1},inplace=True) # This is one way to convert it but we can use another ways and we have the same results.
# Convert the Name and Embarked features to numerical values by using LabelEncoder method:
from sklearn.preprocessing import LabelEncoder 
feature=data["Name"]
feature1=data["Embarked"]
Encoder=LabelEncoder()
data["Embarked"]=Encoder.fit_transform(feature1)
data["Name"]=Encoder.fit_transform(feature)



# Training Data :
from sklearn.model_selection import train_test_split

X=data[["Survived","Pclass","Name","Fare","Age","SibSp","Parch","Embarked"]]
Y=data[["Sex"]]
# convert the data to int64 to be all the columns the same size:
data["Age"]=data["Age"].apply(np.int64)
data["Name"]=data["Name"].apply(np.int64)
data["Fare"]=data["Fare"].apply(np.int64)
data["Embarked"]=data["Embarked"].apply(np.int64)

x_train,x_test,y_train,y_test =train_test_split(X,Y,test_size=0.3,random_state=42 )

#i will use many algorithms to choose the best : 
# Naive Bayes algorithm :
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score , confusion_matrix ,classification_report

gussian=GaussianNB()
gussian.fit(x_train,y_train)
y_pred=gussian.predict(x_test)
print ("The number of all test data : ",len(x_test)) # 268
Accuracy_Naive_bayes=accuracy_score(y_pred,y_test)*100
print ("Accuracy_Naive_bayes:",Accuracy_Naive_bayes) # 71.64 %
print ("Classification_report_Naive_bayes: \n",classification_report(y_pred,y_test)) # for more details like f1-score+ recall +precision.
print ("Confusing Matrix_Naive bayes :\n",confusion_matrix(y_pred,y_test)) # True Female =60 ,True Male=132 

# Support Vector Machine  : 
from sklearn.svm import SVC 
svc_classifier=SVC(kernel="linear")
svc_classifier.fit(x_train,y_train)
y_pred=svc_classifier.predict(x_test)
Accuracy_SVM=accuracy_score(y_pred,y_test)*100
print ("Accuracy_SVM :",Accuracy_SVM) # 79.10 %
print ("Classification_report_SVM: \n",classification_report(y_pred,y_test)) # for more details like f1-score+ recall +precision.
print ("Confusing Matrix_SVM :\n",confusion_matrix(y_pred,y_test)) # True Female =79 ,True Male=133 

# Logistic Regression algorithm:
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
Accuracy_LogisticRegression=accuracy_score(y_pred,y_test)*100
print ("Accuracy_LogisticRegression :",Accuracy_LogisticRegression) # 79.10 %
print ("Classification_report_LogisticRegression: \n",classification_report(y_pred,y_test)) # for more details like f1-score+ recall +precision.
print ("Confusing Matrix_LogisticRegression :\n",confusion_matrix(y_pred,y_test)) # True Female =78 ,True Male=134

#Decision Tree algorithm : 
from sklearn.tree import DecisionTreeClassifier
Tree=DecisionTreeClassifier()
Tree.fit(x_train,y_train)
y_pred=Tree.predict(x_test)
Accuracy_DecisionTree=accuracy_score(y_pred,y_test)*100
print ("Accuracy_DecisionTree :",Accuracy_DecisionTree) # 70.89 %
print ("Classification_report_DecisionTree: \n",classification_report(y_pred,y_test)) # for more details like f1-score+ recall +precision.
print ("Confusing Matrix_DecisionTree :\n",confusion_matrix(y_pred,y_test)) # True Female =58 ,True Male=132

#Random Forests algorithm :
from sklearn.ensemble import RandomForestClassifier
random=RandomForestClassifier(n_estimators=40,random_state=0,min_samples_leaf=1,min_samples_split=10,max_features="auto")
random.fit(x_train,y_train)
y_pred=random.predict(x_test)
Accuracy_RandomForest=accuracy_score(y_pred,y_test)*100
print ("Accuracy_RandomForest :",Accuracy_RandomForest) # 80.59 %
print ("Classification_report_RandomForest: \n",classification_report(y_pred,y_test)) # for more details like f1-score+ recall +precision.
print ("Confusing Matrix_RandomForest :\n",confusion_matrix(y_pred,y_test)) # True Female =72 ,True Male=143

# Which is the best Model ?
conclusion = pd.DataFrame({"Algorithm":["Naive bayes","SVM","Logistic Regression","Decision Tree","Random Forests"],
                          "The Accuracy":[Accuracy_Naive_bayes,Accuracy_SVM,Accuracy_LogisticRegression,Accuracy_DecisionTree,Accuracy_RandomForest]})
result=conclusion.sort_values(by="The Accuracy",ascending=False)
result=result.set_index("The Accuracy")
print (result) # we can see that Random Forests algorithm gave us the greatest Accuracy .

# i will use K-Fold Cross Validation to see the performance of Random Forests algorithm:
# this code will suppose we have K=15, so the result will be an array has 15 different values.
print ("////////////////////////////////////////////")
from sklearn.model_selection import cross_val_score
random_Forest=RandomForestClassifier(n_estimators=40)
score=cross_val_score(random_Forest,x_train,y_train,cv=15,scoring="accuracy")
print ("The Accuracy :",score)
print ("Mean : ",score.mean())
print ("Standard_Deviation: ",score.std())

# To know which the feature has the most effect in RandomForests decision.
importances = pd.DataFrame({'feature':x_train.columns,'importance':np.round(random.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print (importances)
plt.figure()
importances.plot.bar()
plt.savefig("importance.png")