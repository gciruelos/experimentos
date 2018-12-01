Primero clonar el repositorio y entrar al directorio:

    git clone https://github.com/gciruelos/experimentos
    cd experimentos/webcam

Inicializar el virtualenv e instalar requerimientos (primero debe tenerse instalado python3
(y su pip correspondiente), aunque debería andar con python 2 también).

    virtualenv -p python3 venv
    . venv/bin/activate
    pip install -r requirements.txt

Correr:

    python webcam.py mitre.jpeg


Se puede salir apretando `q`.
