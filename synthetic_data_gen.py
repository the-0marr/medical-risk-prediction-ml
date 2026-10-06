
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def sigmoid(x):
    return 1/(1+np.exp(-x))

def generate_synthetic_data():
    np.random.seed(42)
    n_samples=1000
    data=pd.DataFrame({
        'Patient_ID':range(1,n_samples+1),
        'Age':np.random.randint(18,85,size=n_samples),
        'Gender':np.random.choice(['Male','Female'],size=n_samples,p=[0.48,0.52]),
        'Height_cm':np.random.normal(170,10,size=n_samples),
        'Weight_kg':np.random.normal(75,15,size=n_samples),
        'Systolic_BP':np.random.normal(120,20,size=n_samples),
        'Diastolic_BP':np.random.normal(80,15,size=n_samples),
        'Heart_Rate':np.random.normal(70,12,size=n_samples),
        'Temperature_F':np.random.normal(98.6,1.5,size=n_samples),
        'Blood_Sugar':np.random.normal(100,30,size=n_samples),
        'Cholesterol':np.random.normal(200,40,size=n_samples),
        'Hemoglobin':np.random.normal(14,2,size=n_samples),
        'Smoking':np.random.choice(['Yes','No'],size=n_samples,p=[0.25,0.75]),
        'Alcohol_Consumption':np.random.choice(['Yes','No'],size=n_samples,p=[0.30,0.70]),
        'Exercise_Hours_Week':np.random.poisson(3,size=n_samples),
        'Hospital_Visits_Year':np.random.poisson(2,size=n_samples),
        'Insurance_Type':np.random.choice(['Public','Private','Uninsured'],size=n_samples,p=[0.4,0.5,0.1])
    })

    data['Height_cm']=data['Height_cm'].clip(140,210)
    data['Weight_kg']=data['Weight_kg'].clip(40,150)
    data['Systolic_BP']=data['Systolic_BP'].clip(80,220)
    data['Diastolic_BP']=data['Diastolic_BP'].clip(40,140)
    data['Heart_Rate']=data['Heart_Rate'].clip(40,140)
    data['Temperature_F']=data['Temperature_F'].clip(95,105)
    data['Blood_Sugar']=data['Blood_Sugar'].clip(50,300)
    data['Cholesterol']=data['Cholesterol'].clip(100,350)
    data['Hemoglobin']=data['Hemoglobin'].clip(8,20)

    data['BMI']=data['Weight_kg']/((data['Height_cm']/100)**2)

    diabetes_logit=(-7.0+0.045*data['Age']+0.10*(data['BMI']-25)+0.025*(data['Blood_Sugar']-100)-0.12*data['Exercise_Hours_Week']+0.35*(data['Smoking']=='Yes')+0.20*(data['Alcohol_Consumption']=='Yes'))
    diabetes_probability=sigmoid(diabetes_logit)
    data['Diabetes']=np.where(np.random.random(n_samples)<diabetes_probability,'Yes','No')

    hypertension_logit=(-6.0+0.045*data['Age']+0.08*(data['BMI']-25)+0.035*(data['Systolic_BP']-120)+0.025*(data['Diastolic_BP']-80)-0.10*data['Exercise_Hours_Week']+0.30*(data['Smoking']=='Yes'))
    hypertension_probability=sigmoid(hypertension_logit)
    data['Hypertension']=np.where(np.random.random(n_samples)<hypertension_probability,'Yes','No')

    heart_disease_logit=(-8.0+0.055*data['Age']+0.020*(data['Cholesterol']-200)+0.025*(data['Systolic_BP']-120)+0.60*(data['Smoking']=='Yes')+0.70*(data['Diabetes']=='Yes')+0.70*(data['Hypertension']=='Yes')-0.08*data['Exercise_Hours_Week'])
    heart_disease_probability=sigmoid(heart_disease_logit)
    data['Heart_Disease']=np.where(np.random.random(n_samples)<heart_disease_probability,'Yes','No')

    risk_score=(0.30*data['Age']+1.50*data['BMI']+0.10*data['Systolic_BP']+0.08*data['Cholesterol']+0.10*data['Blood_Sugar']+5.0*(data['Smoking']=='Yes')+4.0*(data['Alcohol_Consumption']=='Yes')+8.0*(data['Diabetes']=='Yes')+7.0*(data['Hypertension']=='Yes')+10.0*(data['Heart_Disease']=='Yes')-2.0*data['Exercise_Hours_Week']+np.random.normal(0,3,n_samples))

    min_score=risk_score.min()
    max_score=risk_score.max()
    data['Risk_Score']=((risk_score-min_score)/(max_score-min_score))*100

    data=data[['Patient_ID','Age','Gender','Height_cm','Weight_kg','BMI','Systolic_BP','Diastolic_BP','Heart_Rate','Temperature_F','Blood_Sugar','Cholesterol','Hemoglobin','Smoking','Alcohol_Consumption','Exercise_Hours_Week','Diabetes','Hypertension','Heart_Disease','Hospital_Visits_Year','Insurance_Type','Risk_Score']]

    return data

if __name__=="__main__":
    print("Generating synthetic patient data...")
    clean_data=generate_synthetic_data()
    clean_data.to_csv('clean_patient_data.csv',index=False)
    print(f"Clean dataset generated: {clean_data.shape}")
    print("Saved to: clean_patient_data.csv")
    print("\nFeatures:")
    print(list(clean_data.columns))
    print("\nDiabetes distribution:")
    print(clean_data['Diabetes'].value_counts())
    print("\nHypertension distribution:")
    print(clean_data['Hypertension'].value_counts())
    print("\nHeart Disease distribution:")
    print(clean_data['Heart_Disease'].value_counts())
    print("\nRisk Score statistics:")
    print(clean_data['Risk_Score'].describe())
    numeric_columns=clean_data.select_dtypes(include=[np.number])
    print("\nCorrelation with Risk_Score:")
    print(numeric_columns.corr()['Risk_Score'].sort_values(ascending=False))

