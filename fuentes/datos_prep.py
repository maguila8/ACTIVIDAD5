#funciones para preparar y limpiar los datos

def limpiar_datos(datos):
    #crear una copia para conservar los datos originales
    datos_limpios = datos.copy()

    #conservar las columnas necesarias para la clasificación
    datos_limpios = datos_limpios[["instruction", "category"]].copy()

    #eliminar espacios al inicio y al final de los mensajes
    datos_limpios["instruction"] = (
        datos_limpios["instruction"].str.strip()
    )

    #eliminar registros sin mensaje o sin categoría
    datos_limpios = datos_limpios.dropna(
        subset=["instruction", "category"]
    )

    #eliminar mensajes vacíos
    datos_limpios = datos_limpios[
        datos_limpios["instruction"] != ""
    ]

    #reiniciar los índices
    datos_limpios = datos_limpios.reset_index(drop=True)

    return datos_limpios

