"""Rescata del árbol de git los archivos de PlantDoc que Windows/NTFS no puede materializar con su nombre
original, y verifica que la copia local coincida con el árbol oficial.

Tres causas (clasificación exclusiva; el CSV las separa en `motivo`):
  - `caracter_invalido_ntfs`: el nombre tiene `?` u otro carácter inválido en NTFS (85 archivos);
  - `ruta_demasiado_larga`: el nombre supera MAX_PATH_COMPONENTE caracteres (10; 2 de ellos también tienen `?`);
  - `colision_mayusculas_ntfs`: dos archivos del árbol difieren solo en mayúsculas/minúsculas y NTFS (que no
    distingue) deja uno solo. El rescatado se guarda con el sufijo `__colision_mayusc` (6 archivos).

Qué hace (nunca sobrescribe en silencio; si algo choca, lista los conflictos y no escribe nada):
  1. calcula las rutas faltantes (`git ls-tree` contra el disco) y las escribe en `--salida`;
  2. aplica SIEMPRE el mapeo versionado `--mapeo` (las 101 filas), no solo lo que falte en esta máquina: así el
     resultado no depende de qué rutas largas materialice git en cada PC. Cada archivo se extrae por hash de blob
     (`git cat-file`), no por ruta. Si git materializó el original (rutas largas habilitadas), el original y el
     rescatado serían la misma imagen dos veces: se avisa, y con `--mover-originales` se renombra al nombre local;
  3. verifica el sha1 de cada archivo contra el árbol (y que no sobren ni falten archivos).
`--regenerar` además recalcula el mapeo desde cero y lo compara con el versionado. Lo generado va a `--salida`
(por defecto data/interim/); el CSV versionado no se toca, salvo que se lo pida explícitamente con `--csv-salida`
y `--sobrescribir`.

Uso: python scripts/rescatar_archivos_plantdoc.py --rev <commit> [--repo data/raw/plantdoc] [--regenerar]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path

MAX_PATH_COMPONENTE = 140  # margen bajo el límite de 260 caracteres de Windows para la ruta completa
INVALIDOS_NTFS = '<>:"|?*'
SUFIJO_COLISION = "__colision_mayusc"
MOTIVO_COLISION = "colision_mayusculas_ntfs"
CAMPOS = ["split", "clase", "nombre_original", "nombre_local", "motivo"]
CARPETAS = ("train/", "test/")


# ------------------------------------------------------------------ git
def cargar_mapa_sha(repo: Path, rev: str) -> dict[str, str]:
    """path relativo -> sha1 del blob, vía `git ls-tree -r -z` (no toca el filesystem, evita el límite de longitud
    de ruta de Windows y no entrecomilla nombres raros)."""
    salida = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "-z", rev], capture_output=True, check=True).stdout
    mapa = {}
    for reg in salida.split(b"\0"):
        if not reg:
            continue
        meta, _, path = reg.decode("utf-8").partition("\t")
        mapa[path] = meta.split()[2]
    return mapa


def git_cat_file(repo: Path, sha: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), "cat-file", "blob", sha], capture_output=True, check=True).stdout


def sha1_blob(contenido: bytes) -> str:
    """El hash que git le da a un blob con ese contenido (para comparar contra `ls-tree` sin pedirle nada a git)."""
    return hashlib.sha1(b"blob %d\0" % len(contenido) + contenido).hexdigest()


# ------------------------------------------------------------------ nombres locales
def nombre_local_para(nombre_original: str) -> tuple[str, str]:
    """Devuelve (nombre_local, motivo) para un nombre con `?` o demasiado largo. No decide colisiones de
    mayúsculas (eso depende de los demás archivos de la carpeta; ver `nombre_colision`)."""
    tiene_invalidos = any(c in nombre_original for c in INVALIDOS_NTFS)
    ruta_larga = len(nombre_original) > MAX_PATH_COMPONENTE

    nombre = nombre_original.replace("?", "_")
    if not ruta_larga and len(nombre) <= MAX_PATH_COMPONENTE:
        motivo = "caracter_invalido_ntfs" if tiene_invalidos else "ninguno"
        return nombre, motivo

    stem, punto, ext = nombre.rpartition(".")
    if not punto:
        stem, ext = nombre, ""
    ext = f".{ext}" if ext else ""
    h = hashlib.sha1(nombre_original.encode("utf-8")).hexdigest()[:8]
    recorte = MAX_PATH_COMPONENTE - len(ext) - len(h) - 1
    nombre_corto = f"{stem[:recorte]}_{h}{ext}"
    return nombre_corto, "ruta_demasiado_larga"


def nombre_colision(nombre_original: str) -> str:
    """`Peach-Leaf.jpg` -> `Peach-Leaf__colision_mayusc.jpg`."""
    stem, punto, ext = nombre_original.rpartition(".")
    return f"{stem}{SUFIJO_COLISION}.{ext}" if punto else nombre_original + SUFIJO_COLISION


# ------------------------------------------------------------------ rutas faltantes y mapeo
def rutas_del_arbol(arbol: dict[str, str]) -> list[str]:
    return sorted(p for p in arbol if p.startswith(CARPETAS))


def listar_disco(repo: Path) -> set[str]:
    """Rutas (con `/`) de train/ y test/ tal como las lista el filesystem (con la capitalización guardada)."""
    disco = set()
    for base in ("train", "test"):
        for r, _, fs in os.walk(repo / base):
            for f in fs:
                disco.add(Path(r, f).relative_to(repo).as_posix())
    return disco


def rutas_faltantes(arbol: dict[str, str], disco: set[str]) -> list[str]:
    """Rutas del árbol que no están en el disco con ese nombre exacto (incluye al archivo que NTFS perdió por
    colisión de mayúsculas, porque el disco lista el otro nombre)."""
    return [p for p in rutas_del_arbol(arbol) if p not in disco]


def regenerar_mapeo(faltantes: list[str], arbol: dict[str, str]) -> list[dict]:
    """Mapeo nombre original -> local de lo que falta: colisión de mayúsculas si otro archivo de la misma carpeta
    difiere solo en mayúsculas y el nombre es válido; en otro caso `nombre_local_para`."""
    por_minuscula: dict[str, list[str]] = {}
    for p in rutas_del_arbol(arbol):
        por_minuscula.setdefault(p.lower(), []).append(p)
    filas = []
    for rel in faltantes:
        split, clase, nombre = rel.split("/")
        local, motivo = nombre_local_para(nombre)
        if motivo == "ninguno" and len(por_minuscula[rel.lower()]) > 1:
            local, motivo = nombre_colision(nombre), MOTIVO_COLISION
        filas.append({"split": split, "clase": clase, "nombre_original": nombre, "nombre_local": local, "motivo": motivo})
    return sorted(filas, key=lambda f: (f["motivo"] == MOTIVO_COLISION, f["split"], f["clase"], f["nombre_original"]))


def leer_mapeo(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))
    faltan = [c for c in CAMPOS if filas and c not in filas[0]]
    if faltan:
        raise SystemExit(f"{csv_path}: faltan las columnas {faltan} (¿separador `;`?)")
    return filas


def escribir_csv_mapeo(filas: list[dict], destino: Path, sobrescribir: bool) -> None:
    if destino.exists() and not sobrescribir:
        raise SystemExit(f"{destino} ya existe; no se sobrescribe (usar --sobrescribir si es lo que se quiere)")
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS, lineterminator="\n")
        w.writeheader()
        w.writerows(filas)


# ------------------------------------------------------------------ aplicar el mapeo
def planificar(repo: Path, arbol: dict[str, str], mapeo: list[dict], mover_originales: bool,
               disco: set[str]) -> tuple[list[dict], list[str]]:
    """Qué hacer con cada fila del mapeo y qué conflictos hay. No escribe nada.
    Acciones: `escribir` (extraer el blob), `mover` (renombrar el original ya materializado) o `presente`.
    `disco` = nombres exactos que lista el filesystem: en NTFS `is_file()` ignora mayúsculas y daría "existe" para el
    archivo que una colisión de mayúsculas dejó afuera."""
    acciones, conflictos = [], []
    for fila in mapeo:
        orig = f"{fila['split']}/{fila['clase']}/{fila['nombre_original']}"
        if orig not in arbol:
            conflictos.append(f"{orig}: el mapeo nombra un archivo que no está en el árbol de git")
            continue
        sha = arbol[orig]
        destino = repo / fila["split"] / fila["clase"] / fila["nombre_local"]
        if destino.is_file():
            if sha1_blob(destino.read_bytes()) == sha:
                acciones.append({"accion": "presente", "destino": destino, "sha": sha, "fila": fila})
            else:
                conflictos.append(f"{destino.relative_to(repo).as_posix()}: ya existe con contenido distinto al del árbol; no se sobrescribe")
            continue
        original = repo / orig
        if orig in disco:
            if not mover_originales:
                conflictos.append(f"{orig}: git materializó el original en esta PC (rutas largas habilitadas); quedaría la misma "
                                  f"imagen dos veces. Repetir con --mover-originales para renombrarlo a `{fila['nombre_local']}`")
            elif sha1_blob(original.read_bytes()) != sha:
                conflictos.append(f"{orig}: el original materializado no coincide con el blob del árbol; revisar a mano")
            else:
                acciones.append({"accion": "mover", "origen": original, "destino": destino, "sha": sha, "fila": fila})
            continue
        acciones.append({"accion": "escribir", "destino": destino, "sha": sha, "fila": fila})
    return acciones, conflictos


def aplicar(repo: Path, acciones: list[dict]) -> dict[str, int]:
    n = {"escribir": 0, "mover": 0, "presente": 0}
    for a in acciones:
        n[a["accion"]] += 1
        if a["accion"] == "presente":
            continue
        a["destino"].parent.mkdir(parents=True, exist_ok=True)
        if a["accion"] == "mover":
            a["origen"].rename(a["destino"])
        else:
            contenido = git_cat_file(repo, a["sha"])
            if sha1_blob(contenido) != a["sha"]:
                raise RuntimeError(f"el blob extraído de {a['fila']['nombre_original']} no coincide con su hash")
            with open(a["destino"], "xb") as f:   # "x": falla si el archivo apareció mientras tanto; nunca pisa
                f.write(contenido)
    return n


# ------------------------------------------------------------------ verificación contra el árbol
def verificar(repo: Path, arbol: dict[str, str], mapeo: list[dict]) -> dict:
    """Cada ruta del árbol debe estar en el disco (con su nombre, o con el nombre local del mapeo) con el mismo sha1,
    y no debe haber archivos de más en train/ ni test/."""
    local_de = {f"{f['split']}/{f['clase']}/{f['nombre_original']}": f"{f['split']}/{f['clase']}/{f['nombre_local']}" for f in mapeo}
    faltan, distinto, esperados = [], [], set()
    for rel in rutas_del_arbol(arbol):
        destino = local_de.get(rel, rel)
        esperados.add(destino)
        p = repo / destino
        if not p.is_file():
            faltan.append(rel)
        elif sha1_blob(p.read_bytes()) != arbol[rel]:
            distinto.append(rel)
    disco = listar_disco(repo)
    esperados_min = {e.lower() for e in esperados}
    sobran = sorted(p for p in disco if p not in esperados and p.lower() not in esperados_min)
    return {"esperados": len(esperados), "en_disco": len(disco), "faltan": faltan, "sha1_distinto": distinto, "sobran": sobran,
            "ok": not (faltan or distinto or sobran)}


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="data/raw/plantdoc", help="Ruta al clon local de PlantDoc")
    ap.add_argument("--rev", required=True, help="Commit exacto (ver docs/bitacora/decisiones.md)")
    ap.add_argument("--mapeo", default="docs/bitacora/plantdoc_archivos_renombrados.csv",
                    help="Mapeo versionado nombre original -> local; se aplica entero siempre (101 filas)")
    ap.add_argument("--salida", default="data/interim", help="Carpeta de lo que genera el script (rutas faltantes, mapeo regenerado)")
    ap.add_argument("--regenerar", action="store_true", help="Recalcula el mapeo desde cero y lo compara con el versionado")
    ap.add_argument("--mover-originales", action="store_true", help="Si git materializó un original que el mapeo renombra, lo renombra")
    ap.add_argument("--csv-salida", help="Escribe acá el mapeo regenerado (en vez de en --salida); no pisa un archivo existente")
    ap.add_argument("--sobrescribir", action="store_true", help="Permite que --csv-salida pise un archivo existente")
    ap.add_argument("--sin-verificar", action="store_true", help="No verifica sha1 contra el árbol al final")
    a = ap.parse_args(argv)

    repo, salida = Path(a.repo), Path(a.salida)
    arbol = cargar_mapa_sha(repo, a.rev)
    mapeo = leer_mapeo(Path(a.mapeo))
    print(f"Árbol {a.rev[:8]}: {len(rutas_del_arbol(arbol))} archivos en train/ y test/; mapeo versionado: {len(mapeo)} filas")

    # 1) rutas faltantes (informativo y para regenerar)
    faltantes = rutas_faltantes(arbol, listar_disco(repo))
    salida.mkdir(parents=True, exist_ok=True)
    arch_faltantes = salida / "plantdoc_rutas_faltantes.txt"
    arch_faltantes.write_text("".join(f + "\n" for f in faltantes), encoding="utf-8", newline="\n")
    sin_mapeo = sorted(set(faltantes) - {f"{f['split']}/{f['clase']}/{f['nombre_original']}" for f in mapeo})
    print(f"Rutas faltantes en el disco: {len(faltantes)} (escritas en {arch_faltantes}); sin fila en el mapeo versionado: {len(sin_mapeo)}")
    for r in sin_mapeo[:10]:
        print("  falta y no está en el mapeo:", r)

    # 2) mapeo versionado, siempre entero
    acciones, conflictos = planificar(repo, arbol, mapeo, a.mover_originales, listar_disco(repo))
    if conflictos:
        print(f"\n{len(conflictos)} conflictos; no se escribió nada:", file=sys.stderr)
        for c in conflictos:
            print(" -", c, file=sys.stderr)
        return 1
    n = aplicar(repo, acciones)
    print(f"Mapeo aplicado: {n['escribir']} extraídos del blob, {n['mover']} originales renombrados, {n['presente']} ya presentes y correctos")

    # 3) regenerar y comparar
    rc = 0
    if a.regenerar:
        regenerado = regenerar_mapeo(faltantes, arbol)
        destino = Path(a.csv_salida) if a.csv_salida else salida / "plantdoc_archivos_renombrados.regenerado.csv"
        escribir_csv_mapeo(regenerado, destino, a.sobrescribir)
        clave = lambda f: tuple(f[c] for c in CAMPOS)  # noqa: E731
        igual = {clave(f) for f in regenerado} == {clave(f) for f in mapeo}
        print(f"Mapeo regenerado ({len(regenerado)} filas) en {destino}: {'IGUAL' if igual else 'DISTINTO'} al versionado")
        rc = 0 if igual else 1

    # 4) verificación contra el árbol
    if not a.sin_verificar:
        v = verificar(repo, arbol, mapeo)
        print(f"Verificación sha1 contra el árbol: esperados {v['esperados']}, en disco {v['en_disco']}, faltan {len(v['faltan'])}, "
              f"sha1 distinto {len(v['sha1_distinto'])}, sobran {len(v['sobran'])} -> {'OK' if v['ok'] else 'FALLA'}")
        for k in ("faltan", "sha1_distinto", "sobran"):
            for r in v[k][:10]:
                print(f"  {k}:", r)
        rc = rc or (0 if v["ok"] else 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
