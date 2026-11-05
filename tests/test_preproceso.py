"""Preproceso parametrizado por geometría: una sola función para ambos dominios (invariante 3)."""

import numpy as np
import pytest
import torch
from PIL import Image

from foliares.data.preproceso import GEOMETRIAS, MEDIA_IMAGENET, STD_IMAGENET, preprocesar


def _imagen(w, h, color=(200, 30, 30)):
    return Image.new("RGB", (w, h), color)


@pytest.mark.parametrize("geometria", GEOMETRIAS)
@pytest.mark.parametrize("w,h", [(256, 256), (300, 120), (90, 400)])
def test_forma_y_tipo_para_cualquier_imagen(geometria, w, h):
    x = preprocesar(_imagen(w, h), geometria, 224)
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32


def test_normalizacion_imagenet():
    x = preprocesar(_imagen(10, 10, (255, 0, 0)), "aplastar", 8)
    esperado = (torch.tensor([1.0, 0.0, 0.0]) - torch.tensor(MEDIA_IMAGENET)) / torch.tensor(STD_IMAGENET)
    assert torch.allclose(x[:, 4, 4], esperado, atol=1e-5)


def test_en_una_imagen_cuadrada_aplastar_y_recortar_coinciden():
    gen = np.random.default_rng(0)
    im = Image.fromarray(gen.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    assert torch.equal(preprocesar(im, "aplastar", 224), preprocesar(im, "resize_recorte", 224))


def test_relleno_conserva_la_proporcion_y_rellena_con_la_media():
    x = preprocesar(_imagen(400, 100), "relleno", 224)        # queda 224×56 centrado: 84 filas de relleno arriba y abajo
    assert torch.allclose(x[:, :80, :], torch.zeros(3, 80, 224), atol=0.03)   # media de ImageNet normalizada ≈ 0
    assert not torch.allclose(x[:, 112, :], torch.zeros(3, 224), atol=0.03)


def test_resize_recorte_toma_el_centro():
    im = Image.new("RGB", (300, 100), (0, 0, 255))
    im.paste((255, 0, 0), (100, 0, 200, 100))                  # franja central roja de 100 px
    x = preprocesar(im, "resize_recorte", 100)                 # lado corto a 100 => recorte central = la franja roja
    rojo = (torch.tensor([1.0, 0.0, 0.0]) - torch.tensor(MEDIA_IMAGENET)) / torch.tensor(STD_IMAGENET)
    assert torch.allclose(x[:, 50, 50], rojo, atol=1e-5)


def test_geometria_desconocida_falla():
    with pytest.raises(ValueError, match="Geometría desconocida"):
        preprocesar(_imagen(10, 10), "estirar", 8)


def test_acepta_ruta(tmp_path):
    _imagen(50, 40).save(tmp_path / "a.png")
    assert torch.equal(preprocesar(tmp_path / "a.png", "relleno", 32), preprocesar(_imagen(50, 40), "relleno", 32))


def test_aplica_la_orientacion_exif(tmp_path):
    # Imagen de 40×20 guardada con orientación EXIF 6 (rotar 90°): leída, tiene que quedar de 20×40.
    im = Image.new("RGB", (40, 20), (0, 0, 0))
    im.paste((255, 255, 255), (0, 0, 20, 20))                                  # mitad izquierda blanca
    exif = Image.Exif()
    exif[0x0112] = 6
    im.save(tmp_path / "rotada.jpg", exif=exif.tobytes(), quality=100)
    from foliares.data.preproceso import leer_rgb
    with Image.open(tmp_path / "rotada.jpg") as abierta:
        assert leer_rgb(abierta).size == (20, 40)


def test_el_alfa_se_compone_sobre_blanco():
    from foliares.data.preproceso import leer_rgb
    transparente = Image.new("RGBA", (4, 4), (255, 0, 0, 0))                   # rojo totalmente transparente
    assert leer_rgb(transparente).getpixel((0, 0)) == (255, 255, 255)
    medio = Image.new("RGBA", (4, 4), (0, 0, 0, 128))                          # negro al 50 %
    assert all(abs(c - 127) <= 1 for c in leer_rgb(medio).getpixel((0, 0)))


def test_una_imagen_rgb_sin_exif_no_cambia():
    from foliares.data.preproceso import leer_rgb
    gen = np.random.default_rng(1)
    im = Image.fromarray(gen.integers(0, 255, (30, 20, 3), dtype=np.uint8))
    assert np.array_equal(np.asarray(leer_rgb(im)), np.asarray(im))
