"""App en Gradio (etapa 6, sesión 12): sube una foto de una hoja o sácala con la cámara y muestra el top-3 con probabilidad
calibrada y el CAM. El modelo, T y τ salen de configs/05_calibracion.yaml (vía configs/06_app.yaml), sin cambiarlos.

Uso (desde la raíz del repo):
  python app/app.py --config configs/06_app.yaml --salida <carpeta con los checkpoints, fuera del repo>

Todo lo que no es interfaz (preproceso, inferencia, calibración, abstención, CAM) está en foliares.inferencia.
"""

import argparse
from pathlib import Path

import gradio as gr
import yaml

from foliares.eval.evaluacion import _nombres, cargar_modelo_base
from foliares.eval.figuras import nombre_corto
from foliares.inferencia import predecir_imagen
from foliares.utils.paths import cargar_rutas

CARTEL = "Confianza baja: no usar como diagnóstico"

# El 97 % es la exactitud en PV-val (0,971) de results/06_app/06_app_pv_val_20261110_093958/pv_val_informativo.json.
AVISO = """**Alcance y uso.** Esta app es una orientación, **no un diagnóstico**: ante cualquier duda, consultá a un técnico.
Cubre tomate, papa y pimiento, y solo estas {n} clases: {clases}.

La confianza se calibró con fotos de campo, donde el modelo acierta poco. Por eso es baja incluso cuando la respuesta es correcta, \
y el cartel de confianza baja aparece en casi todas las imágenes. El orden del top-3 sigue siendo informativo: con fotos de \
laboratorio, la primera opción acierta el 97 %."""


def crear_responder(modelo, nombres, temperatura, tau, geometria, tamano):
    """Función que recibe la ruta de la imagen y devuelve (CAM superpuesto, top-3 para gr.Label, cartel en Markdown)."""

    def responder(ruta):
        if ruta is None:
            raise gr.Error("Subí una foto o sacá una con la cámara.")
        try:
            r = predecir_imagen(modelo, ruta, nombres, temperatura, tau, geometria, tamano)
        except OSError as e:                      # PIL.UnidentifiedImageError es un OSError
            raise gr.Error(f"No se pudo leer la imagen: {e}") from e
        top = {nombre_corto(c): p for c, p in r["top"]}
        cartel = (f'<div style="background:#c0262d;color:#fff;padding:10px;border-radius:6px;font-weight:600">{CARTEL}</div>'
                  if r["baja_confianza"] else "")
        return r["cam"], top, cartel

    return responder


def construir_demo(responder, nombres, temperatura, tau) -> gr.Blocks:
    with gr.Blocks(title="Enfermedades foliares") as demo:
        gr.Markdown("# Enfermedades foliares: tomate, papa y pimiento")
        gr.Markdown(AVISO.format(n=len(nombres), clases="; ".join(nombre_corto(n) for n in nombres)))
        with gr.Row():
            entrada = gr.Image(label="Foto de la hoja", sources=["upload", "webcam"], type="filepath")
            with gr.Column():
                cartel = gr.Markdown()
                top = gr.Label(label=f"Top-3 (probabilidad calibrada, T = {temperatura:.2f})", num_top_classes=3)
                cam = gr.Image(label="CAM de la primera opción (dónde mira el modelo; escala relativa)", interactive=False)
        gr.Markdown(f"Umbral de confianza τ = {tau}: por debajo, se muestra el cartel.")
        entrada.change(responder, entrada, [cam, top, cartel])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/06_app.yaml"))
    ap.add_argument("--salida", required=True, type=Path, help="carpeta fuera del repo donde está el checkpoint de la config")
    ap.add_argument("--puerto", type=int, default=7860)
    a = ap.parse_args()
    R = cargar_rutas()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    cal = yaml.safe_load((R.repo / cfg["calibracion"]).read_text(encoding="utf-8"))
    nombres, _, _ = _nombres(yaml.safe_load((R.repo / cal["config_modelo"]).read_text(encoding="utf-8")))
    if cal.get("temperatura") is None:
        raise SystemExit("configs/05_calibracion.yaml no tiene `temperatura` fijada")
    modelo = cargar_modelo_base(a.salida, cal, len(nombres))   # verifica el sha256 del checkpoint
    responder = crear_responder(modelo, nombres, cal["temperatura"], cal["tau"], cal["geometria"], cal["tamano"])
    construir_demo(responder, nombres, cal["temperatura"], cal["tau"]).launch(server_port=a.puerto)


if __name__ == "__main__":
    main()
