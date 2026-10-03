import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def clean_workplace_data(filepath: str) -> pd.DataFrame:
    """
    Cleans and standardizes survey.csv (Workplace Mental Health Data)
    """
    df = pd.read_csv(filepath)
    
    # Clean Age Outliers
    df['Age'] = df['Age'].apply(lambda x: np.nan if (x < 18 or x > 80) else x)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Standardize Messy Gender Strings
    def standardize_gender(g):
        if pd.isna(g): 
            return 'Other'
        g_str = str(g).lower().strip()
        if g_str in ['male', 'm', 'male-ish', 'maile', 'cis male', 'mal', 'male (cis)', 'make', 'guy (-ish) ^_^', 'male ', 'man', 'msle', 'mail', 'cis man', 'malr']:
            return 'Male'
        elif g_str in ['female', 'f', 'cis female', 'woman', 'femake', 'female ', 'cis-female/femme', 'female (cis)', 'femail']:
            return 'Female'
        return 'Other'
        
    df['Gender'] = df['Gender'].apply(standardize_gender)
    
    # Clean Target Column (treatment -> 0/1)
    df['target'] = df['treatment'].map({'Yes': 1, 'No': 0})
    
    return df


def clean_student_data(filepath: str) -> pd.DataFrame:
    """
    Cleans and prepares Student Mental Health Analysis During Online Learning.csv
    """
    df = pd.read_csv(filepath)
    
    # Clean column whitespace
    df.columns = df.columns.str.strip()
    
    # Target Encoding (High Stress = 1, Medium/Low = 0)
    df['target'] = df['Stress Level'].map({'Low': 0, 'Medium': 0, 'High': 1})
    
    return df