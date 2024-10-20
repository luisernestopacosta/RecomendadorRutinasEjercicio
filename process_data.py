import pandas as ps

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

    X = df.drop("Precio", axis=1)
    y = df["Precio"]

    return X,y
