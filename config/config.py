from dataclasses import dataclass

@dataclass
class Paths:
    procesados: str
    limpios: str
    crudos: str
    
@dataclass
class Preprocesado:
    lw: bool
    pc: bool
    st: bool
    tk: bool

@dataclass
class Mostrado:
    matriz: bool
    grafos: bool
    estudio: bool
    
@dataclass
class Modelos:
    modelo: str

@dataclass
class Pipeline:
    preprocesado: bool
    balanceo: bool
    entrenamiento: bool
    testeo: bool

@dataclass
class GlobalConfig:
    paths: Paths
    preprocesado: Preprocesado
    mostrado: Mostrado
    modelos: Modelos
    pipeline: Pipeline