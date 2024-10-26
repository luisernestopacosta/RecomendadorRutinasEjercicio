from model_training import training
from sklearn.metrics import mean_squared_error
import questionary

# Tipo Combustible,
# Marca,
# Edicion,
# Color,
# Tipo de Vehiculo,
# Estatus de Vehiculo,

prediction = model.predict(X_test)
print(X_test)
mse = mean_squared_error(Y_test, prediction)

print(f"Prediction: {prediction}")
print(f"Mean Squared Error: {mse}")