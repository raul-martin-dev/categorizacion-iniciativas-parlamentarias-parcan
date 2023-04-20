import sys
sys.path.append('..')
sys.path.append('../..')

from data.preprocesar_data import preprocesar_data
from models.regresion_logistica import regresion_logistica
from models.validacion_cruzada import validacion_cruzada

import hydra
from hydra.core.config_store import ConfigStore
from config.config import GlobalConfig
cs = ConfigStore.instance() 
cs.store(name='nlp_config', node=GlobalConfig)


@hydra.main(version_base=None, config_path='../../config', config_name='config')
def pipeline(cfg: GlobalConfig):
    print("\n> Categorización de iniciativas parlamentarias")
    print("Autor: Raúl Martín Rigor\n")
    
    if cfg.pipeline.preprocesado == True:
        preprocesar_data(cfg)

    if cfg.pipeline.entrenamiento == True:
        if cfg.modelos.modelo == 'logistica':
            regresion_logistica(cfg)
        if cfg.modelos.modelo == 'validacion':
            validacion_cruzada(cfg)
            

if __name__ == '__main__':
    pipeline()
    
