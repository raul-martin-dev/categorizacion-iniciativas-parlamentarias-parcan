import os
import subprocess

# Directorios principales
config = 'config'
data = 'data'
docs = 'docs'
models = 'models'
reports = 'reports'
scripts = 'scripts'
source = 'source'

# Creación de los directorios principales
os.mkdir(config)
os.mkdir(data)
os.mkdir(docs)
os.mkdir(models)
os.mkdir(reports)
os.mkdir(scripts)
os.mkdir(source)

# Directorio data (carpetas)
clean = 'clean'
external = 'external'
interim = 'interim'
processed = 'processed'
raw = 'raw'
test = 'test'

# Creación de los directorios dentro de data
path = os.path.join(f'{data}/', clean)
os.mkdir(path)
path = os.path.join(f'{data}/', external)
os.mkdir(path)
path = os.path.join(f'{data}/', interim)
os.mkdir(path)
path = os.path.join(f'{data}/', processed)
os.mkdir(path)
path = os.path.join(f'{data}/', raw)
os.mkdir(path)
path = os.path.join(f'{data}/', test)
os.mkdir(path)

# Directorios de source (carpetas)
data = 'data'
features = 'features'
methods = 'methods'
models = 'models'
tests = 'tests'

# Creación de los directorios dentro de source
path = os.path.join(f'{source}/', data)
os.mkdir(path)
path = os.path.join(f'{source}/', features)
os.mkdir(path)
path = os.path.join(f'{source}/', methods)
os.mkdir(path)
path = os.path.join(f'{source}/', models)
os.mkdir(path)
path = os.path.join(f'{source}/', tests)
os.mkdir(path)

# Creación de los ficheros de config
config_py = f'{config}/config.py'
config_yaml = f'{config}/config.yaml'

try:
    open(config_py, 'a').close()
    open(config_yaml, 'a').close()
except OSError:
    print('Failed creating the config files')
else:
    print('Files created')

# Creación del fichero de documentación
documentation = f'{docs}/documentation.md'

try:
    open(documentation, 'a').close()
except OSError:
    print('Failed creating the doc file')
else:
    print('Doc file created')

# Creación de los ficheros del programa principal
git = '.gitignore'
make = 'Makefile'
readme = 'README.md'

try:
    open(git, 'a').close()
    open(make, 'a').close()
    open(readme, 'a').close()
except OSError:
    print('Failed creating the main files')
else:
    print('Main files created')

# Creación del fichero de requerimientos
command = "pip freeze > requirements.txt"
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, shell=True)
#Launch the shell command:
output = process.communicate()[0]