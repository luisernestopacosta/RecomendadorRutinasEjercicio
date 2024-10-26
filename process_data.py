import pandas as ps
from sklearn.preprocessing import LabelEncoder 

df = ps.read_csv("data/vehiculos.csv")

def load_data():
    global df
    # print(df.head(10))
    # print(df.dtypes)
    # print(df.isnull().sum())

    df = df[(df["Kilometros"] >= 0) | ((df["Kilometros"] <= 1000) & (df["Estatus de Vehiculo"] == "Nuevo"))]
    
    df["Tipo Combustible"] = df["Tipo Combustible"].apply(lambda x: "GLP" if x == "Gas" else x)

    df = df[df['Numero de Puertas'].isin([2, 4])]
    df = df.drop("Modelo", axis=1)    

    label_encoder = LabelEncoder()
    df['Tipo Combustible']= label_encoder.fit_transform(df['Tipo Combustible']) 

    df['Marca']= label_encoder.fit_transform(df['Marca']) 

    df['Edicion']= label_encoder.fit_transform(df['Edicion']) 

    df['Color']= label_encoder.fit_transform(df['Color'])

    df['Tipo de Vehiculo']= label_encoder.fit_transform(df['Tipo de Vehiculo'])

    df['Estatus de Vehiculo'] = label_encoder.fit_transform(df['Estatus de Vehiculo'])

    X = df.drop("Precio", axis=1)
    y = df["Precio"]

    return X,y
