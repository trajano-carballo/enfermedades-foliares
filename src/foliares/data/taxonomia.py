"""Taxonomías de referencia para el inventario de PlantVillage y PlantDoc.

No implementa ninguna decisión de mapeo: solo agrupa las clases oficiales de cada
dataset por cultivo, para poder filtrar y contar. El mapeo clase-a-clase candidato
(con sus ambigüedades) vive en docs/mapeo_clases.md, no acá.
"""

from __future__ import annotations

CLASES_PLANTVILLAGE = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Cultivo (crop) de cada clase de PlantVillage, tal como aparece en el prefijo
# antes de "___". Se usa para filtrar por cultivo en el inventario y en la
# verificación de leaf_grouping (tarea 3 de la sesión 2).
CULTIVO_POR_CLASE_PLANTVILLAGE = {
    c: c.split("___")[0] for c in CLASES_PLANTVILLAGE
}

CULTIVOS_CANDIDATOS = {
    "maiz": "Corn_(maize)",
    "tomate": "Tomato",
    "papa": "Potato",
    "pimiento": "Pepper,_bell",
}


def clases_plantvillage_de_cultivo(cultivo: str) -> list[str]:
    """Clases de PlantVillage cuyo prefijo de cultivo coincide con `cultivo`
    (una de las claves de CULTIVOS_CANDIDATOS, ej. 'maiz', 'tomate')."""
    prefijo = CULTIVOS_CANDIDATOS[cultivo]
    return [c for c in CLASES_PLANTVILLAGE if c.split("___")[0] == prefijo]


# Clases candidatas de PlantDoc por cultivo (tomate, papa, pimiento), a partir
# del inventario oficial train/test (ver notebooks/00_exploracion_datasets.ipynb).
# Es un listado descriptivo de qué existe en el dataset, no una decisión de
# inclusión/exclusión: eso lo marca docs/mapeo_clases.md.
CLASES_PLANTDOC_CANDIDATAS = {
    "tomate": [
        "Tomato Early blight leaf",
        "Tomato Septoria leaf spot",
        "Tomato leaf",
        "Tomato leaf bacterial spot",
        "Tomato leaf late blight",
        "Tomato leaf mosaic virus",
        "Tomato leaf yellow virus",
        "Tomato mold leaf",
        "Tomato two spotted spider mites leaf",
    ],
    "papa": [
        "Potato leaf early blight",
        "Potato leaf late blight",
    ],
    "pimiento": [
        "Bell_pepper leaf",
        "Bell_pepper leaf spot",
    ],
}

CLASES_PLANTDOC_MAIZ = [
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
]
