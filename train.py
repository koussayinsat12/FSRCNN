from model import Model
import yaml
if __name__ =='__main__':
    config_path='config.yaml'
    model=Model(config_path)
    model.train()


   