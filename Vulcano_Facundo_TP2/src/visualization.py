import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_histograms(features: list, df):
    for feature in features:
        df[feature].hist(bins=30, figsize=(15, 10), edgecolor='black')
        plt.suptitle(f'Histograma de {feature}')
        plt.xlabel(f'{feature}')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        plt.show()

def scatter_plot(df, x_feature, y_feature, group_feature):
    if y_feature == 'Kilómetros':
        if df[y_feature].dtype == 'object':
            df[y_feature] = df[y_feature].str.replace(' km', '').str.replace('.', '').astype(float)
    df[x_feature] = df[x_feature].astype(float)
    df[y_feature] = df[y_feature].astype(float)
    plt.figure(figsize=(10, 6))
    for type_ in df[group_feature].unique():
        subset = df[df[group_feature] == type_]
        plt.scatter(subset[x_feature], subset[y_feature], label=type_, alpha=0.7)

    plt.title(f"Scatter plot de {y_feature} vs {x_feature} por {group_feature} de vehiculo")
    plt.xlabel(f"{x_feature}")
    plt.ylabel(f"{y_feature}")
    plt.legend(title="Tipo de vehiculo")
    plt.grid(True)
    plt.show()

def group_motor(motor_df):
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


def grouped_histograms(feature1, feature2, df):
    df_grouped = df.groupby([feature1, feature2]).size().unstack(fill_value=0)
    df_grouped.plot(kind='bar', stacked=True, figsize=(10, 6))

    plt.title(f'Distribucion de {feature1} segun {feature2} de vehiculo')
    plt.xlabel(feature1)
    plt.ylabel('Cantidad de vehiculo')
    plt.legend(title=feature2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def histograms_with_price(df, feature1, feature2):
    df_grouped = df.groupby([feature2, feature1])['Precio'].mean().unstack(fill_value=0)
    df_grouped.plot(kind='bar', stacked=False, figsize=(10, 6))

    plt.title(f'Precio Promedio de Vehículos según {feature1}')
    plt.xlabel(feature1)
    plt.ylabel('Precio Promedio')
    plt.legend(title=feature2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

df = pd.read_csv('C:\\Users\\facuv\\Machine-Learning\\Vulcano_Facundo_TP2\\data\\raw\\toyota_dev.csv')




