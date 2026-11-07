"""Etapa 4 (sesión 10): aumentación solo en train, descongelado parcial con BatchNorm congelada y entrenamiento reanudable."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from foliares.data.aumentacion import transformacion_train
from foliares.data.congelar import sha256_archivo
from foliares.data.preproceso import preprocesar
from foliares.models.mobilenet import MobileNetV3SmallGAP, descongelar_parcial, modo_entrenamiento_parcial
from foliares.training import finetune as FT
from foliares.training.checkpoint import guardar_checkpoint

REPO = Path(__file__).resolve().parents[1]
AUG = {"tipo": "dirigida", "escala_recorte": [0.6, 1.0], "rotacion_grados": 20, "brillo": 0.3, "contraste": 0.3, "saturacion": 0.3}


@pytest.fixture(autouse=True)
def restaurar_modos_globales():
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    yield
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn


@pytest.fixture
def imagen(tmp_path):
    g = torch.Generator().manual_seed(0)
    ruta = tmp_path / "hoja.png"
    Image.fromarray((torch.rand(80, 60, 3, generator=g) * 255).byte().numpy()).save(ruta)
    return ruta


# ------------------------------------------------------------------ tarea 1: aumentación
def test_con_aumentacion_ninguna_el_pipeline_de_train_es_el_del_base(imagen):
    f = transformacion_train({"tipo": "ninguna"}, "aplastar", 64)
    assert torch.equal(f(imagen), preprocesar(imagen, "aplastar", 64))


def test_la_aumentacion_dirigida_cambia_la_imagen_y_es_reproducible_con_la_semilla(imagen):
    f = transformacion_train(AUG, "aplastar", 64)
    torch.manual_seed(1)
    a = f(imagen)
    torch.manual_seed(1)
    b = f(imagen)
    torch.manual_seed(2)
    c = f(imagen)
    assert a.shape == (3, 64, 64) and torch.equal(a, b) and not torch.equal(a, c)
    assert not torch.equal(a, preprocesar(imagen, "aplastar", 64))


def test_aumentacion_desconocida_falla():
    with pytest.raises(ValueError, match="Aumentación desconocida"):
        transformacion_train({"tipo": "mixup"}, "aplastar", 64)


def test_la_config_real_aumenta_solo_train_y_evalua_con_aplastar():
    import yaml
    cfg = yaml.safe_load((REPO / "configs" / "04_aumentacion.yaml").read_text(encoding="utf-8"))
    assert cfg["aumentacion"]["tipo"] == "dirigida" and cfg["preproceso"] == {"geometria": "aplastar", "tamano": 224}
    assert cfg["finetune"]["bloques_entrenables"] == [11, 12] and cfg["finetune"]["lr"] == 1e-4 and cfg["finetune"]["batch"] == 64
    assert (cfg["finetune"]["max_epocas"], cfg["finetune"]["paciencia"]) == (20, 5) and cfg["semilla"] == 42 and cfg["determinista"]


# ------------------------------------------------------------------ tarea 2: descongelado parcial
def test_solo_los_ultimos_dos_bloques_y_fc_tienen_gradiente():
    m = descongelar_parcial(MobileNetV3SmallGAP(12), [11, 12])
    con = {n.split(".")[0] + ("." + n.split(".")[1] if n.startswith("features") else "") for n, p in m.named_parameters() if p.requires_grad}
    assert con == {"features.11", "features.12", "fc"}
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 294096 + 56448 + 12 * 576 + 12


def test_tras_un_paso_la_batchnorm_de_un_bloque_congelado_no_cambia():
    torch.manual_seed(0)
    m = descongelar_parcial(MobileNetV3SmallGAP(4), [11, 12])
    bn_congelada = m.features[5].block[0][1]                # BatchNorm de un bloque congelado
    bn_entrenable = m.features[11].block[0][1]
    antes = [t.clone() for t in (bn_congelada.running_mean, bn_congelada.running_var, bn_entrenable.running_mean)]
    peso_congelado = m.features[5].block[0][0].weight.clone()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    modo_entrenamiento_parcial(m, [11, 12])
    assert not m.features[5].training and m.features[11].training and m.fc.training
    F.cross_entropy(m(torch.randn(4, 3, 64, 64)), torch.tensor([0, 1, 2, 3])).backward()
    opt.step()
    assert torch.equal(bn_congelada.running_mean, antes[0]) and torch.equal(bn_congelada.running_var, antes[1])
    assert not torch.equal(bn_entrenable.running_mean, antes[2])          # la de un bloque entrenable sí se actualiza
    assert torch.equal(m.features[5].block[0][0].weight, peso_congelado)


# ------------------------------------------------------------------ tarea 3: entrenamiento
def _cfg(**ft):
    return {"experimento": "04_test", "semilla": 42, "determinista": True,
            "modelo_base": {"run_id": "base", "sha256": None}, "preproceso": {"geometria": "aplastar", "tamano": 64},
            "aumentacion": AUG, "finetune": {"bloques_entrenables": [11, 12], "optimizador": "adam", "lr": 1e-3, "batch": 8,
                                            "max_epocas": 3, "paciencia": 5, "criterio": "macro_f1_val_con_mosaic", "workers": 0, **ft}}


CFG_MODELO = {"clases": [{"pv": "Clase___A", "plantdoc": "a"}, {"pv": "Clase___B", "plantdoc": "b"}], "clase_mosaic": "Clase___B",
              "metricas": {"modo_sin_mosaic": "sin_imagenes_de_la_clase", "desempate_trivial": "primera_en_orden"}}


@pytest.fixture
def base(tmp_path):
    torch.manual_seed(0)
    m = MobileNetV3SmallGAP(2)
    ruta = tmp_path / "base.pt"
    guardar_checkpoint(ruta, m, torch.optim.SGD(m.parameters(), lr=0.1), 0)
    return ruta


def _entrenar(tmp_path, pv_sintetico, base, cfg, nombre):
    splits, data_root = pv_sintetico
    cfg = {**cfg, "modelo_base": {**cfg["modelo_base"], "sha256": sha256_archivo(base)}}
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return FT.entrenar(cfg, CFG_MODELO, repo, splits, data_root, tmp_path / nombre / "res", tmp_path / nombre / "ck", base, "r1")


def test_entrena_escribe_el_json_y_solo_toca_plantvillage(tmp_path, pv_sintetico, base):
    d = _entrenar(tmp_path, pv_sintetico, base, _cfg(), "a")
    for k in ("config", "commit", "cambios_sin_commitear", "sha256_splits", "semillas", "torch", "cuda_runtime", "gpu", "dispositivo",
              "tiempo_s", "memoria_pico_cpu_mb", "memoria_pico_gpu_mb", "curva", "epoca_elegida", "pv_val", "checkpoint_mejor"):
        assert k in d, k
    assert d["n_train"] == 24 and d["n_val"] == 6 and d["semillas"]["determinista"] is True and len(d["curva"]) == 3
    assert (tmp_path / "a" / "ck" / "04_test" / "r1" / "mejor.pt").is_file()
    assert (tmp_path / "a" / "res" / "04_test" / "r1" / "resultados.json").is_file()


def test_un_checkpoint_base_con_otro_sha256_se_rechaza(tmp_path, pv_sintetico, base):
    splits, data_root = pv_sintetico
    cfg = {**_cfg(), "modelo_base": {"run_id": "base", "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="sha256"):
        FT.entrenar(cfg, CFG_MODELO, tmp_path, splits, data_root, tmp_path / "res", tmp_path / "ck", base, "r1")


def test_retomar_despues_de_un_corte_da_lo_mismo_que_la_corrida_continua(tmp_path, pv_sintetico, base, monkeypatch):
    continua = _entrenar(tmp_path, pv_sintetico, base, _cfg(), "continua")
    real, llamadas = FT.predecir_val, []

    def corta_en_la_tercera_epoca(*a, **k):
        llamadas.append(1)
        if len(llamadas) == 3:
            raise RuntimeError("sesión de Colab cortada")
        return real(*a, **k)
    monkeypatch.setattr(FT, "predecir_val", corta_en_la_tercera_epoca)
    with pytest.raises(RuntimeError, match="cortada"):
        _entrenar(tmp_path, pv_sintetico, base, _cfg(), "cortada")
    monkeypatch.setattr(FT, "predecir_val", real)
    retomada = _entrenar(tmp_path, pv_sintetico, base, _cfg(), "cortada")
    assert retomada["reanudaciones_desde_epoca"] == [2]
    sin_t = lambda c: [{k: v for k, v in e.items() if k != "tiempo_s"} for e in c]  # noqa: E731
    assert sin_t(retomada["curva"]) == sin_t(continua["curva"]) and retomada["epoca_elegida"] == continua["epoca_elegida"]
    assert retomada["checkpoint_mejor"]["sha256"] != "" and retomada["pv_val"] == continua["pv_val"]


def test_no_retoma_con_otra_config(tmp_path, pv_sintetico, base):
    _entrenar(tmp_path, pv_sintetico, base, _cfg(max_epocas=1), "x")
    with pytest.raises(ValueError, match="otra config"):
        _entrenar(tmp_path, pv_sintetico, base, _cfg(max_epocas=2), "x")


def test_paciencia_detiene(tmp_path, pv_sintetico, base):
    d = _entrenar(tmp_path, pv_sintetico, base, _cfg(max_epocas=10, paciencia=1, lr=0.0), "p")    # LR 0: nunca mejora
    assert d["detenido_por"] == "paciencia" and d["epoca_elegida"] == 1 and d["epocas_corridas"] == 2


def test_el_script_exige_salida_ckpt_y_base(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "entrenar_finetune.py"), "--config", str(REPO / "configs" / "04_aumentacion.yaml")],
                       capture_output=True, text=True, cwd=REPO, env={**os.environ, "DATA_ROOT": str(tmp_path)})
    assert r.returncode == 2 and all(s in r.stderr for s in ("--salida", "--ckpt-dir", "--checkpoint-base"))


@pytest.mark.parametrize("aug", [AUG, {"tipo": "ninguna"}])
def test_los_datasets_se_pueden_serializar_para_los_workers(pv_sintetico, aug):
    # Con workers > 0 y spawn (Windows), el DataLoader manda el Dataset por pickle: una lambda o una función interna fallan.
    import pickle
    from foliares.data.dataset import DatasetParticion
    splits, data_root = pv_sintetico
    tr = DatasetParticion(splits, data_root, "plantvillage", "train", "entrenamiento", ["Clase___A", "Clase___B"],
                          preproceso=transformacion_train(aug, "aplastar", 64))
    va = DatasetParticion(splits, data_root, "plantvillage", "val", "validacion", ["Clase___A", "Clase___B"], geometria="aplastar", tamano=64)
    for d in (tr, va):
        copia = pickle.loads(pickle.dumps(d))
        torch.manual_seed(3)
        a = copia[0][0]
        torch.manual_seed(3)
        assert torch.equal(a, d[0][0])


def test_el_notebook_de_colab_entrena_con_la_config_y_no_lleva_plantdoc():
    import json
    celdas = ["".join(c["source"]) for c in json.loads((REPO / "notebooks" / "04_aumentacion_colab.ipynb").read_text(encoding="utf-8"))["cells"]
              if c["cell_type"] == "code"]
    todo = "\n".join(celdas)
    assert 'for ds in ("plantvillage",):' in todo and "plantdoc_subconjunto" not in todo        # PlantDoc no entra a Colab
    assert "configs/04_aumentacion.yaml" in todo and all(f in todo for f in ("--checkpoint-base", "--ckpt-dir", "--salida", "--run-id"))
    assert 'REPO_REF  = "main"  # @param' in todo and "RUN_ID_RETOMAR" in todo
