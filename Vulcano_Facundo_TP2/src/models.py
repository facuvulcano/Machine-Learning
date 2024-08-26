import numpy as np
import pandas as pd

class RideRegression():
    def __init__(self, features, targets) -> None:
        self.features = np.array(features)
        self.targets = np.array(targets)
        self.features_traspose = self.features.T

    def target_prediction(self, weights):
        y = np.dot(self.features, weights)
        return y
    
    def optimal_weights(self):
        #Uso de pseudoinversa en lugar de inversa
        XTX = self.features_traspose.dot(self.features)
        if np.linalg.det(XTX) == 0:
            return np.linalg.pinv(XTX).dot(self.features_traspose).dot(self.targets)
        else:   
            return np.linalg.inv(XTX).dot(self.features_traspose).dot(self.targets)



class NonLinearRegression():
    pass

class LocallyWeightedRegression():
    pass


df_procesado = pd.read_csv('C:\\Users\\facuv\\Machine-Learning\\Vulcano_Facundo_TP2\\data\\processed\\dataset_procesado.csv')

targets = df_procesado['Precio'].to_numpy()
df_procesado = df_procesado.drop(columns=['Precio'])
features = df_procesado.to_numpy()


regresion = RideRegression(features, targets)
weights = regresion.optimal_weights()
print(f'predicciones de precios: {regresion.target_prediction(weights)}')
print(f'precios reales: {targets}')

