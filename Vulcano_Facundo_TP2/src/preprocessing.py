import pandas as pd
import os
from datetime import datetime

def one_hot_encoder(df, feature):
    one_hot = pd.get_dummies(df[feature])
    return one_hot
    
def normalize(X, feature, min_val, max_val, rango):

    min_max_scaler = rango[0] + ((X[feature] - min_val) / (max_val - min_val)) * (rango[1] - rango[0])
    return min_max_scaler

def handle_missing_values(df, feature):
    cars_without_feature = df[df[feature].isna()]

    mean_price_hilux = cars_without_feature[cars_without_feature['Tipo'] == 'Hilux SW4']['Precio'].mean()
    mean_price_corolla = cars_without_feature[cars_without_feature['Tipo'] == 'Corolla Cross']['Precio'].mean()
    mean_price_RAV4 = cars_without_feature[cars_without_feature['Tipo'] == 'RAV4']['Precio'].mean()

    hilux_with_color = df[(df['Tipo'] == 'Hilux SW4') & (df[feature].notna())]
    corolla_with_color = df[(df['Tipo'] == 'Corolla Cross') & (df[feature].notna())]
    RAV4_with_color = df[(df['Tipo'] == 'RAV4') & (df[feature].notna())]

    hillux_dict = {}
    hillux_smallest = 1000000
    for color in hilux_with_color['Color'].unique():
        mean_price_color = hilux_with_color[hilux_with_color['Color'] == color]['Precio'].mean()
        hillux_diff = abs(mean_price_hilux - mean_price_color)
        hillux_dict[hillux_diff] = color
        if hillux_diff < hillux_smallest:
            hillux_smallest = hillux_diff

    corolla_dict = {}
    corolla_smallest = 1000000
    for color in corolla_with_color['Color'].unique():
        mean_price_color = corolla_with_color[corolla_with_color['Color'] == color]['Precio'].mean()
        corolla_diff = abs(mean_price_corolla - mean_price_color)
        corolla_dict[corolla_diff] = color
        if corolla_diff < corolla_smallest:
            corolla_smallest = corolla_diff
    
    RAV4_dict = {}
    RAV4_smallest = 1000000
    for color in RAV4_with_color['Color'].unique():
        mean_price_color = RAV4_with_color[RAV4_with_color['Color'] == color]['Precio'].mean()
        RAV4_diff = abs(mean_price_RAV4 - mean_price_color)
        RAV4_dict[RAV4_diff] = color
        if RAV4_diff < RAV4_smallest:
            RAV4_smallest = RAV4_diff

    df.loc[(df['Tipo'] == 'Hilux SW4') & (df['Color'].isna()), 'Color'] = hillux_dict[hillux_smallest]
    df.loc[(df['Tipo'] == 'Corolla Cross') & (df['Color'].isna()), 'Color'] = corolla_dict[corolla_smallest]
    df.loc[(df['Tipo'] == 'RAV4') & (df['Color'].isna()), 'Color'] = RAV4_dict[RAV4_smallest]

def group_by_engine(motor_df):
    agrupaciones = {
        '2.8' : ['2.8', '2,8', '2.8 tdi', '2.8 204 cv', '2.8 td 204cv', '2.8tdi c/tgv 204cv y cadena distr', '2.8 204cv toyota'],
        '3.0' : ['3.0', '3'],
        '2.0' : ['2.0', '2', '2.0 cvt'],
        '2.7' : ['2.7'],
        '1.8' : ['1.8', '1.8l'],
        '2.5' : ['2.5', '2.5 awd'],
        '2.4' : ['2.4'],
        '1' : ['1'],
        'hibrido' : ['hibrido', '1.8 hibrido', '18hv', 'nafta / hibrido', '1.8 nafta combinado con uno eléctrico'],
        'otros' : ['inyeccion multi punto', '-', '222 cv', 'toyota', 'ecvt', 'toyota 1gd', 'srx', 
                   '4 cilindros en línea con turbocompresor de geometría variable (tgv) e intercooler', 'srv sw4', 'nan',
                   'diesel', 'diesel td', 'nafta']        
    }
    for grupo, valores in agrupaciones.items():
        for valor in valores:
            if valor == motor_df:
                return grupo
    return 'otros'

def processing(df):
    df['Año'] = df['Año'].astype(int)
    df['Kilómetros'] = df['Kilómetros'].str.replace(' km', '').str.replace('.', '').astype(int)
    df['Motor'] = df['Motor'].apply(group_by_engine)
    handle_missing_values(df, 'Color')

    df = df.drop(columns=['id'])
    df = df.drop(columns=['Tipo de vendedor'])
    df = df.drop(columns=['Color'])

    features_to_encode = ['Tipo', 'Tipo de combustible', 'Transmisión', 'Motor']
    for feature in features_to_encode:
        feature_one_hot_encoded = one_hot_encoder(df, feature).astype(int)
        df = pd.concat([df, feature_one_hot_encoded], axis=1)
        df = df.drop(columns=[feature])

    df = df.drop(columns=['Híbrido/Nafta'])
    df = df.drop(columns=['Automática secuencial'])    
    df = df.drop(columns=['otros'])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"dataset_procesado_{timestamp}.csv"
    output_path = os.path.join('/home/facuvulcano/Machine-Learning/Vulcano_Facundo_TP2/data/processed', file_name)

    df.to_csv(output_path, index=False)
    print(f"Archivo guardado como: {output_path}")

    return output_path