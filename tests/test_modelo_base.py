"""Invariantes del modelo base y corrida de punta a punta con un PlantVillage sintético (sin red, sin datos reales)."""

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from foliares import train
from foliares.data.dataset import DatasetParticion, ParticionInvalida
from foliares.data.taxonomia import CLASES_EVAL_PV, PAR_CANDIDATO_PLANTDOC
from foliares.models.mobilenet import MobileNetV3SmallGAP
from foliares.training.features import congelar_backbone

REPO = Path(__file__).resolve().parents[1]
CFG_REAL = yaml.safe_load((REPO / "configs" / "02_modelo_base.yaml").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def restaurar_modos_globales():
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


def test_config_orden_alfabetico_de_las_12_y_mapeo_a_plantdoc():
    pv = [c["pv"] for c in CFG_REAL["clases"]]
    assert pv == sorted(CLASES_EVAL_PV) and len(pv) == 12
    assert all(c["plantdoc"] == PAR_CANDIDATO_PLANTDOC[c["pv"]] for c in CFG_REAL["clases"])
    assert CFG_REAL["clase_mosaic"] in pv


def test_un_solo_linear_576_a_12_sin_capa_oculta():
    m = MobileNetV3SmallGAP(12)
    lineales = [x for x in m.modules() if isinstance(x, nn.Linear)]
    assert len(lineales) == 1 and (lineales[0].in_features, lineales[0].out_features) == (576, 12)


def test_con_backbone_congelado_solo_el_linear_es_entrenable():
    m = congelar_backbone(MobileNetV3SmallGAP(12))
    entrenables = {n for n, p in m.named_parameters() if p.requires_grad}
    assert entrenables == {"fc.weight", "fc.bias"}


@pytest.mark.parametrize("rol", ["entrenamiento", "validacion"])
def test_train_y_val_no_se_pueden_construir_con_plantdoc(pv_sintetico, rol):
    splits, data_root = pv_sintetico
    with pytest.raises(ParticionInvalida, match="invariante 1"):
        DatasetParticion(splits, data_root, "plantdoc", "dev", rol, ["Clase___A", "Clase___B"], ["a", "b"])


def _features(semilla=0, n=60, d=576, k=3):
    g = torch.Generator().manual_seed(semilla)
    y = torch.arange(n) % k
    return torch.randn(n, d, generator=g) + y[:, None].float(), y


ENT = {"optimizador": "adam", "lr": 1e-3, "batch": 16, "max_epocas": 8, "paciencia": 3}


def test_misma_semilla_mismo_resultado_sobre_features():
    X, y = _features()
    a, b = nn.Linear(576, 3), nn.Linear(576, 3)
    ra = train.entrenar_cabezal(a, X, y, X, y, ENT, 42, True, 3)
    rb = train.entrenar_cabezal(b, X, y, X, y, ENT, 42, True, 3)
    assert [{k: v for k, v in c.items() if k != "tiempo_s"} for c in ra["curva"]] == \
           [{k: v for k, v in c.items() if k != "tiempo_s"} for c in rb["curva"]]
    assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias) and ra["epoca_elegida"] == rb["epoca_elegida"]


def test_early_stopping_por_paciencia_y_queda_la_mejor_epoca():
    X, y = _features()
    fc = nn.Linear(576, 3)
    r = train.entrenar_cabezal(fc, X, y, X, y, {**ENT, "max_epocas": 100, "paciencia": 2}, 42, True, 3)
    mejor = max(c["macro_f1_val"] for c in r["curva"])
    primera_mejor = next(c["epoca"] for c in r["curva"] if c["macro_f1_val"] == mejor)
    assert r["epoca_elegida"] == primera_mejor and r["macro_f1_val_elegida"] == mejor
    assert r["detenido_por"] == "paciencia" and r["epocas_corridas"] == primera_mejor + 2


def test_control_de_desbalance():
    assert train.control_desbalance({"a": 1500, "b": 262}, 5)["excede"]
    assert not train.control_desbalance({"a": 500, "b": 100}, 5)["excede"]


def _cfg(**extra):
    cfg = {**CFG_REAL, "clases": [{"pv": "Clase___A", "plantdoc": "a"}, {"pv": "Clase___B", "plantdoc": "b"}],
           "clase_mosaic": "Clase___B", "preproceso": {"geometria": "aplastar", "tamano": 32},
           "modelo": {"pesos_imagenet": False}, "features": {"batch": 8},
           "entrenamiento": {**CFG_REAL["entrenamiento"], "max_epocas": 4, "paciencia": 2},
           "metricas": {"modo_sin_mosaic": "sin_imagenes_de_la_clase", "desempate_trivial": "primera_en_orden"},
           "desbalance_aceptado_por_el_equipo": False}
    cfg.update(extra)
    return cfg


def _ejecutar(tmp_path, pv_sintetico, cfg, limite=None):
    splits, data_root = pv_sintetico
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return train.ejecutar(cfg, repo, splits, data_root, tmp_path / "salida", tmp_path / "cache", "r1", limite), repo


def test_corrida_de_punta_a_punta_escribe_lo_que_exige_el_bloque_comun(tmp_path, pv_sintetico):
    d, repo = _ejecutar(tmp_path, pv_sintetico, _cfg())
    salida = tmp_path / "salida" / "02_modelo_base" / "r1"
    escrito = json.loads((salida / "resultados.json").read_text(encoding="utf-8"))
    for k in ("config", "commit", "cambios_sin_commitear", "sha256_splits", "semillas", "torch", "dispositivo",
              "tiempo_extraccion_s", "tiempo_entrenamiento_s", "memoria_extraccion", "memoria_entrenamiento",
              "conteo_por_clase", "pv_val", "trivial_pv_val", "checkpoint"):
        assert k in escrito, k
    assert escrito["semillas"]["determinista"] is True and escrito["entrenamiento"]["curva"]
    assert "macro_f1_sin_Clase___B" in escrito["pv_val"] and (salida / "mejor_checkpoint.pt").is_file()
    copiados = sorted(p.name for p in (repo / "results" / "02_modelo_base" / "r1").iterdir())
    assert "resultados.json" in copiados and all(not n.endswith(".pt") for n in copiados) and len(copiados) == 3


def test_la_segunda_corrida_usa_el_cache_de_features(tmp_path, pv_sintetico):
    _ejecutar(tmp_path, pv_sintetico, _cfg())
    d, _ = _ejecutar(tmp_path, pv_sintetico, _cfg())
    assert all(f["desde_cache"] for f in d["features"].values())


def test_frena_antes_de_entrenar_si_el_desbalance_supera_el_umbral(tmp_path, pv_sintetico):
    with pytest.raises(train.DesbalanceExcedido, match="se frena antes de entrenar"):
        _ejecutar(tmp_path, pv_sintetico, _cfg(umbral_desbalance=0.5))
    assert not (tmp_path / "salida").exists() and not (tmp_path / "cache").exists()     # no escribió nada


def test_con_el_desbalance_aceptado_entrena_y_lo_deja_registrado(tmp_path, pv_sintetico):
    d, _ = _ejecutar(tmp_path, pv_sintetico, _cfg(umbral_desbalance=0.5, desbalance_aceptado_por_el_equipo=True))
    assert d["control_desbalance"]["excede"] and d["control_desbalance"]["aceptado_por_el_equipo"]
    assert d["entrenamiento"]["epoca_elegida"] >= 1


def test_la_config_real_trae_las_decisiones_del_equipo():
    assert CFG_REAL["desbalance_aceptado_por_el_equipo"] is True
    assert CFG_REAL["metricas"] == {"modo_sin_mosaic": "sin_imagenes_de_la_clase", "desempate_trivial": "primera_en_orden"}


def test_exige_fijar_las_definiciones_de_metricas_antes_de_calcular(tmp_path, pv_sintetico):
    cfg = _cfg()
    del cfg["metricas"]
    with pytest.raises(ValueError, match="modo_sin_mosaic"):
        _ejecutar(tmp_path, pv_sintetico, cfg)


def test_la_salida_tiene_que_estar_fuera_del_repo(tmp_path, pv_sintetico):
    splits, data_root = pv_sintetico
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="fuera del repo"):
        train.ejecutar(_cfg(), repo, splits, data_root, repo / "results", tmp_path / "cache", "r1")


def test_la_prueba_corta_no_copia_nada_al_repo(tmp_path, pv_sintetico):
    d, repo = _ejecutar(tmp_path, pv_sintetico, _cfg(), limite=4)
    assert d["prueba_corta"] and not (repo / "results").exists()


def test_la_clave_del_cache_incluye_el_codigo_del_preproceso_y_del_modelo_y_pil():
    # M2: si cambia preproceso.py, mobilenet.py o la versión de PIL, las features cacheadas no se reutilizan.
    from foliares.training.features import clave_features
    clave = clave_features("plantvillage", "train", ["a"], [0], "x", "aplastar", 224, False, "cpu")
    assert {"sha256_preproceso_py", "sha256_mobilenet_py", "pil"} <= set(clave)


@pytest.mark.parametrize("campo,valor", [("criterio", "perdida_val"), ("aumentacion", "flip"), ("ponderacion_clases", True)])
def test_ejecutar_falla_si_la_config_pide_algo_no_implementado(tmp_path, pv_sintetico, campo, valor):
    # M3: el código solo implementa macro_f1_val_con_mosaic, sin aumentación y sin ponderación.
    cfg = _cfg()
    if campo == "criterio":
        cfg["entrenamiento"] = {**cfg["entrenamiento"], "criterio": valor}
    else:
        cfg[campo] = valor
    with pytest.raises(ValueError, match="no está implementad"):
        _ejecutar(tmp_path, pv_sintetico, cfg)
    assert not (tmp_path / "salida").exists()


def test_el_json_usa_rutas_relativas_y_aclara_la_memoria(tmp_path, pv_sintetico):
    d, _ = _ejecutar(tmp_path, pv_sintetico, _cfg())
    rutas = [d["checkpoint"]["ruta"]] + [f["archivo"] for f in d["features"].values()]
    assert all(not Path(r).is_absolute() and str(tmp_path) not in r for r in rutas), rutas
    assert "proceso" in d["nota_memoria"]
