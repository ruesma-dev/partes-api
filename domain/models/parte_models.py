# domain/models/parte_models.py
"""Esquema de extraccion de un PARTE DE TRABAJO (sv2).

Modelo real del parte de Construcciones Ruesma (una pagina = un parte):
  - Cabecera: UN dia (fecha), UNA obra (numero + nombre/ubicacion),
    el encargado y el jefe de obra.
  - Una tabla de PERSONAL con HASTA 14 filas; cada fila con datos es un
    empleado: categoria, nombre, horas ordinarias y horas
    extraordinarias. Si en vez de (o ademas de) horas hay una INCIDENCIA
    (vacaciones, baja, etc.), se captura normalizada al codigo de la
    leyenda del parte.
  - Una "ASIGNACION DEL PERSONAL POR PARTIDAS DEL PRESUPUESTO": reparto
    opcional de horas por partida del presupuesto, por empleado.
  - FIRMA: el parte lleva un Vo Bo / firma; se confirma si esta firmado
    y en que recuadro (Encargado / Jefe de Obra / Control / Administracion)
    y, si es legible, el nombre.

La leyenda de incidencias del parte (RELACION DE INCIDENCIAS) es FIJA y
la conoce el sistema; la IA NO la lee, solo NORMALIZA lo que aparezca al
codigo correspondiente:
    V  = Vacaciones
    B  = Baja Enfermedad Comun / No Profesional
    AT = Baja por Accidente de Trabajo
    FJ = Faltas Justificadas (permisos / licencias)
    F  = Faltas No Justificadas
    H  = Huelgas
    M  = Maternidad / Paternidad

El CODIGO DE HORA de Sigrid (auxhor) NO lo decide la IA: lo resuelve sv3
a partir de horas ordinarias/extraordinarias (auxhor.ext) o de la
incidencia. El portal (sv4) permite cambiarlo desde la lista de Sigrid.

``extra="forbid"`` (StrictSchemaModel): si la IA emite un campo no
declarado, el parseo falla en vez de tragarse datos silenciosamente.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from domain.models.schema_base import StrictSchemaModel


class IncidenciaParte(StrictSchemaModel):
    """Incidencia de un empleado en el dia del parte (si la hay)."""

    # Codigo NORMALIZADO a la leyenda: V|B|AT|FJ|F|H|M. None si no aplica.
    codigo: Optional[str] = None
    # Texto literal tal cual aparece ('V', 'vacaciones', 'vacaci', 'baja'...).
    texto_leido: Optional[str] = None
    # Numero de dias de la incidencia si el parte lo indica.
    dias: Optional[float] = None
    # CODIGO de Sigrid (auxhor) PROPUESTO por la IA para esta incidencia,
    # elegido del catalogo inyectado (p.ej. 'CIV' para vacaciones). None si
    # no se aporto catalogo o la IA no encuentra uno adecuado.
    codigo_sigrid: Optional[str] = None


class PartidaAsignacion(StrictSchemaModel):
    """Reparto de horas a una partida del presupuesto (seccion derecha)."""

    partida: Optional[str] = None  # codigo o descripcion de la partida
    horas: Optional[float] = None


class EmpleadoParte(StrictSchemaModel):
    # Numero de fila del PERSONAL (1..14) tal cual figura en el parte.
    numero_linea: Optional[int] = None
    categoria: Optional[str] = None        # Capataz, Oficial 1a, Peon...
    nombre: Optional[str] = None
    # DNI/NIE de la columna "DNI" (plantillas J.310 rev. 1+; impreso o
    # manuscrito). None si el parte no tiene esa columna o esta vacia.
    dni: Optional[str] = None

    horas_ordinarias: Optional[float] = None
    horas_extraordinarias: Optional[float] = None

    # CODIGOS de hora de Sigrid (auxhor) PROPUESTOS por la IA segun la
    # CATEGORIA del empleado y el tipo (normal/extra), elegidos del catalogo
    # inyectado en el prompt. None si no se aporto catalogo. Ej.: para un
    # Capataz -> ordinaria 'MCAP' (Mes Capataz), extra 'HECAP' (Hora Extra
    # Capataz).
    codigo_hora_ordinaria: Optional[str] = None
    codigo_hora_extra: Optional[str] = None

    # Incidencia (vacaciones/baja/etc.) si en lugar de/ademas de horas se
    # anota una. None si el empleado solo tiene horas normales.
    incidencia: Optional[IncidenciaParte] = None

    # Reparto por partidas del presupuesto, si la seccion derecha tiene
    # datos para este empleado.
    partidas: List[PartidaAsignacion] = Field(default_factory=list)

    confianza_pct: Optional[float] = Field(default=None, ge=0, le=100)


class FirmaParte(StrictSchemaModel):
    # Hay alguna firma en el parte? (cualquiera de las 3 casillas).
    firmado: bool = False
    # Estado de CADA UNA de las tres casillas de firma del pie del parte.
    # True si ESA casilla concreta tiene una firma/rubrica manuscrita.
    firma_encargado: bool = False
    firma_jefe_obra: bool = False
    firma_administracion: bool = False
    # Recuadro principal donde aparece la firma y nombre si es legible.
    firmante_rol: Optional[str] = None
    firmante_nombre: Optional[str] = None
    confianza_pct: Optional[float] = Field(default=None, ge=0, le=100)


class CabeceraParte(StrictSchemaModel):
    # Fecha del parte (un parte = un dia). ISO YYYY-MM-DD.
    fecha: Optional[str] = None
    # Numero de obra tal cual ('672'); sv3 lo normaliza al codigo Sigrid.
    obra_numero: Optional[str] = None
    obra_nombre: Optional[str] = None       # nombre + ubicacion de la obra
    encargado_nombre: Optional[str] = None
    jefe_obra_nombre: Optional[str] = None


class ParteTrabajo(StrictSchemaModel):
    cabecera: CabeceraParte
    firma: FirmaParte
    empleados: List[EmpleadoParte]
