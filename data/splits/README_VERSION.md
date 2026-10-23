# Particiones congeladas — versión v1

- Fecha: 2026-10-23
- Commit del repo al congelar: `b79d3a271811b347e9401c9cebb85fae7fe7a322` (cambios sin commitear: no)
- Datasets (commits fijados): plantvillage `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`, plantdoc `5467f6012d78d1c446145d5f582da6096f852ae8`
- Estado de la revisión humana de etiquetas (PlantDoc): COMPLETO

## Archivos y sha256

- `plantdoc_particion.csv`: `dc82e2d5af43a91610a0342023f34a972561fe143e8dfe13c6d4cc02768c8bba`
- `plantvillage_particion.csv`: `1ff04397bed8182b901609f2e0d3e0e8b3ceed3ac4b7739512b64f9b366713dd`

## Conteos por partición

- plantvillage: 22787 filas — excluida: 3210, test: 2446, train: 14648, val: 2483
- plantdoc: 1100 filas — (sin partición): 104, dev: 696, test: 300

Metadatos de generación (semilla, zona, tope, fracciones):

```json
{
  "plantvillage": {
    "semilla": 42,
    "zona_mosaic": 20,
    "tope_train_por_clase": 1500,
    "fracciones": [
      0.7,
      0.15,
      0.15
    ],
    "commit_plantvillage": "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a",
    "clases_eval": 12,
    "conteos_12_clases": {
      "excluida": 3210,
      "test": 2170,
      "train": 11968,
      "val": 2207
    },
    "script": "notebooks/01_preparacion_datos.ipynb",
    "commit_repo": "a6423b5a4fd7f3ae45e9ea610a0af743464adcca",
    "repo_con_cambios_sin_commitear": false
  },
  "plantdoc": {
    "semilla": 42,
    "frac_test": 0.3,
    "umbral_duplicado": 6,
    "commit_plantdoc": "5467f6012d78d1c446145d5f582da6096f852ae8",
    "revision_modo": "COMPLETO",
    "revision_estado": "COMPLETO",
    "revision_pendiente": [],
    "revision": {
      "planilla": {
        "estado": "COMPLETA",
        "completa": true,
        "filas": 124,
        "regla": "se excluye si revisor_1 = descartar; revisor_2 se ignora (D1, 2026-10-09)",
        "decisiones": {
          "revisor_1": {
            "mantener": 114,
            "descartar": 10
          }
        },
        "filas_invalidas": 0,
        "descartar_sin_motivo": [],
        "revisor_ignorado_con_valores": {},
        "avisos": []
      },
      "pares_7_10": {
        "estado": "COMPLETO",
        "completo": true,
        "pares": 3,
        "decisiones": {
          "misma_foto": 3
        },
        "avisos": [],
        "regla": "misma_foto => se une a los grupos de duplicados antes del split (D3, 2026-10-09)"
      }
    },
    "estados": {
      "conservada": 996,
      "excluida_contradictoria": 62,
      "excluida_duplicada": 30,
      "excluida_revision": 10,
      "fuera_de_alcance": 2
    },
    "exclusiones_por_clase": {
      "Bell_pepper leaf": {
        "excluida_revision": 3
      },
      "Bell_pepper leaf spot": {
        "excluida_duplicada": 3,
        "excluida_contradictoria": 1
      },
      "Potato leaf early blight": {
        "excluida_contradictoria": 18,
        "excluida_duplicada": 3
      },
      "Potato leaf late blight": {
        "excluida_contradictoria": 12,
        "excluida_duplicada": 4
      },
      "Tomato Early blight leaf": {
        "excluida_contradictoria": 10,
        "excluida_duplicada": 3
      },
      "Tomato Septoria leaf spot": {
        "excluida_contradictoria": 7,
        "excluida_duplicada": 6
      },
      "Tomato leaf": {
        "excluida_revision": 7
      },
      "Tomato leaf bacterial spot": {
        "excluida_duplicada": 4,
        "excluida_contradictoria": 3
      },
      "Tomato leaf late blight": {
        "excluida_contradictoria": 7,
        "excluida_duplicada": 5
      },
      "Tomato leaf mosaic virus": {
        "excluida_contradictoria": 1
      },
      "Tomato leaf yellow virus": {
        "excluida_duplicada": 2,
        "excluida_contradictoria": 2
      },
      "Tomato mold leaf": {
        "excluida_contradictoria": 1
      }
    },
    "exclusiones_revision_por_motivo": {
      "1": 7,
      "3": 2,
      "2": 1
    },
    "pares_7_10_aplicados": [
      {
        "ruta_a": "raw/plantdoc/train/Tomato Early blight leaf/early-blight-or-target-spot-alternaria-solani-lesions-on-a-tomato-BY9J8R.jpg",
        "ruta_b": "raw/plantdoc/train/Potato leaf early blight/early-blight-or-target-spot-alternaria-solani-lesions-on-a-tomato-by9j8r.jpg",
        "clase_a": "Tomato Early blight leaf",
        "clase_b": "Potato leaf early blight",
        "estado_a": "excluida_contradictoria",
        "estado_b": "excluida_contradictoria"
      },
      {
        "ruta_a": "raw/plantdoc/train/Potato leaf early blight/80109626.jpg",
        "ruta_b": "raw/plantdoc/train/Potato leaf early blight/potato-early-blight-alternaria-alternata-lesions-on-potato-leaf-XDF64E.jpg",
        "clase_a": "Potato leaf early blight",
        "clase_b": "Potato leaf early blight",
        "estado_a": "conservada",
        "estado_b": "excluida_duplicada"
      },
      {
        "ruta_a": "raw/plantdoc/train/Potato leaf late blight/B2650062-Potato_late_blight-SPL.jpg",
        "ruta_b": "raw/plantdoc/train/Potato leaf late blight/potato-late-blight-phytophthora-infestans-necrotic-disease-lesions-B7DEW3.jpg",
        "clase_a": "Potato leaf late blight",
        "clase_b": "Potato leaf late blight",
        "estado_a": "conservada",
        "estado_b": "excluida_duplicada"
      }
    ],
    "commit_repo": "a6423b5a4fd7f3ae45e9ea610a0af743464adcca",
    "repo_con_cambios_sin_commitear": false,
    "script": "notebooks/01b_depuracion_plantdoc.ipynb"
  }
}
```

Estos archivos NO se regeneran tras la etapa 1. Un cambio requiere decisión del equipo y nueva versión.
