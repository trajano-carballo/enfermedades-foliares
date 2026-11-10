"""Sesión 14: cada número de docs/trazabilidad.md coincide con su fuente (JSON de results/ o CSV congelado de data/splits/).

Tabla de fuentes: | `ALIAS` | `ruta` | config | `commit` | accesos |. Tabla de números: | ID | número | valor | `ALIAS` | `clave` |.
Clave de un JSON (o de un YAML de configs/): ruta con puntos (`resultados.PD-test.n`; un segmento numérico indexa una lista); varias claves con `;`.
Clave de un CSV: condiciones `col=valor` o `col!=valor` unidas con `&`; el valor es el número de filas que las cumplen.
Las claves que empiezan con `derivado` o `—` no se verifican acá (la fórmula está escrita en la tabla).
"""

import json
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "trazabilidad.md"


def _celdas(linea: str) -> list[str]:
    return [c.strip() for c in linea.strip().strip("|").split("|")]


def _sin_tildes(c: str) -> str:
    return c.strip("`")


def tablas():
    fuentes, numeros = {}, []
    for linea in DOC.read_text(encoding="utf-8").splitlines():
        if not linea.startswith("|"):
            continue
        c = _celdas(linea)
        if len(c) == 5 and re.fullmatch(r"`[A-Z0-9_]+`", c[0]) and c[1].startswith("`"):
            fuentes[_sin_tildes(c[0])] = {"ruta": _sin_tildes(c[1]), "commit": _sin_tildes(c[3])}
        elif len(c) == 5 and re.fullmatch(r"[A-Z]\d+", c[0]):
            numeros.append({"id": c[0], "valor": c[2], "alias": _sin_tildes(c[3]), "clave": _sin_tildes(c[4])})
    return fuentes, numeros


def numeros_de(texto: str) -> list[tuple[float, int]]:
    """'0,807 [0,768; 0,845]' -> [(0.807, 3), ...]; '2.170' -> [(2170, 0)] (punto = miles, coma = decimales)."""
    out = []
    for t in re.findall(r"-?\d+(?:[.,]\d+)*", texto):
        entero, _, dec = t.replace(".", "").partition(",")
        out.append((float(f"{entero}.{dec}" if dec else entero), len(dec)))
    return out


def valores_json(d: dict, clave: str) -> list[float]:
    out = []
    for c in clave.split(";"):
        v = d
        for seg in c.strip().split("."):
            v = v[int(seg)] if isinstance(v, list) else v[seg]
        out += [v["estimacion"], v["ic_bajo"], v["ic_alto"]] if isinstance(v, dict) else [v]
    return out


def valor_csv(ruta: Path, clave: str) -> list[float]:
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    m = pd.Series(True, index=df.index)
    for cond in clave.split("&"):
        col, op, val = re.fullmatch(r"(\w+)(!=|=)(.*)", cond.strip()).groups()
        m &= (df[col] == val) if op == "=" else (df[col] != val)
    return [float(m.sum())]


FUENTES, NUMEROS = tablas()


def test_la_tabla_tiene_fuentes_y_numeros():
    assert len(FUENTES) >= 15 and len(NUMEROS) >= 50
    assert len({n["id"] for n in NUMEROS}) == len(NUMEROS), "IDs repetidos"
    assert {n["alias"] for n in NUMEROS} <= set(FUENTES), "alias sin fuente"


@pytest.mark.parametrize("fuente", [f for f in FUENTES.values() if f["ruta"].endswith(".json")], ids=lambda f: f["ruta"])
def test_commit_de_cada_json(fuente):
    d = json.loads((REPO / fuente["ruta"]).read_text(encoding="utf-8"))
    assert d["commit"].startswith(fuente["commit"]) and d["cambios_sin_commitear"] is False


@pytest.mark.parametrize("n", [n for n in NUMEROS if not n["clave"].startswith(("derivado", "—"))], ids=lambda n: n["id"])
def test_cada_numero_coincide_con_su_fuente(n):
    ruta = REPO / FUENTES[n["alias"]]["ruta"]
    if ruta.suffix == ".csv":
        reales = valor_csv(ruta, n["clave"])
    else:                                  # JSON de results/ o YAML de configs/ (yaml lee los dos)
        reales = valores_json(yaml.safe_load(ruta.read_text(encoding="utf-8")), n["clave"])
    escritos = numeros_de(n["valor"])
    assert len(escritos) == len(reales), f"{n['id']}: {n['valor']!r} contra {reales}"
    for (x, dec), real in zip(escritos, reales):
        assert abs(real - x) <= 0.5 * 10 ** -dec + 1e-9, f"{n['id']}: escrito {x} (con {dec} decimales), fuente {real}"


def test_cada_id_citado_en_el_informe_tiene_fila():
    informe = (REPO / "docs" / "informe_final.md").read_text(encoding="utf-8")
    citados = set(re.findall(r"\b[A-Z]\d+\b", " ".join(re.findall(r"<!--(.*?)-->", informe, re.S))))
    assert citados and citados <= {n["id"] for n in NUMEROS}, sorted(citados - {n["id"] for n in NUMEROS})


def ids_sin_su_numero(informe: str, numeros: list[dict], ventana: int = 700) -> list[str]:
    """IDs citados en un comentario cuyo valor (ningún número de la celda Valor) no aparece en los `ventana` caracteres
    anteriores al comentario, sin contar otros comentarios. Detecta un ID pegado a un número que no es el suyo."""
    valor = {n["id"]: n["valor"] for n in numeros}
    malos = []
    for m in re.finditer(r"<!--(.*?)-->", informe, re.S):
        antes = re.sub(r"<!--.*?-->", " ", informe[max(0, m.start() - ventana):m.start()], flags=re.S)
        antes = re.sub(r"^.*?-->", " ", antes, flags=re.S)          # comentario cortado por el borde de la ventana
        for i in re.findall(r"\b[A-Z]\d+\b", m.group(1)):
            tokens = re.findall(r"\d+(?:[.,]\d+)*", valor[i])
            if not any(re.search(rf"(?<![\d.,]){re.escape(t)}(?!\d|[.,]\d)", antes) for t in tokens):
                malos.append(f"{i} (línea {informe.count(chr(10), 0, m.start()) + 1})")
    return malos


def test_cada_id_tiene_su_numero_cerca():
    assert ids_sin_su_numero((REPO / "docs" / "informe_final.md").read_text(encoding="utf-8"), NUMEROS) == []


def test_ids_sin_su_numero_detecta_un_id_mal_puesto():
    numeros = [{"id": "E9", "valor": "0,807 [0,768; 0,845]"}, {"id": "E8", "valor": "0,130 [0,093; 0,168]"}]
    assert ids_sin_su_numero("la brecha es 0,807 <!-- E9 -->", numeros) == []
    assert ids_sin_su_numero("la brecha es 0,8071 <!-- E9 -->", numeros) == ["E9 (línea 1)"]       # no vale un prefijo
    assert ids_sin_su_numero("PD-test 0,130 <!-- E8 --> y la brecha <!-- E9 -->", numeros) == ["E9 (línea 1)"]
