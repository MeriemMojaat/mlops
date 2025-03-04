# Définition des variables
VENV=venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip
IMAGE_NAME=meriem_mojaat_4ds1_mlops
VERSION=latest
CONTAINER_NAME=fastapi_mlflow_container
DOCKER_HUB_USER=meriemmojaat

# Installation des dépendances
install:
	if [ ! -d "$(VENV)" ]; then python3 -m venv $(VENV); fi
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "🔧 Dependencies installed."

# Vérification du code (formatage, qualité, sécurité)
lint:
	$(PIP) install flake8 bandit
	flake8 --max-line-length=88 --ignore=E203,W503,SyntaxWarning *.py
	bandit -r . --exclude ./__pycache__,./tests,./venv
	@echo "🔍 Code linted."

format:
	black .
	@echo "📝 Code formatted."

# Préparation des données
prepare:
	$(PYTHON) main.py --train_data churn-bigml-80.csv --test_data churn-bigml-20.csv --prepare_data
	@echo "🔄 Data prepared."

# Entraînement du modèle (with save)
train:
	$(PYTHON) main.py --train_data churn-bigml-80.csv --test_data churn-bigml-20.csv --train --save
	@echo "📈 Model trained and saved."

# Évaluation du modèle
evaluate:
	$(PYTHON) main.py --train_data churn-bigml-80.csv --test_data churn-bigml-20.csv --evaluate
	@echo "📊 Model evaluated."

# Lancer l'API avec FastAPI
run_api:
	$(PYTHON) -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
	@echo "🌐 FastAPI running on port 8000."

# Lancer l'interface MLflow
start_mlflow:
	mlflow ui --host 0.0.0.0 --port 5000 &
	@echo "💻 MLflow UI started on port 5000."

# Construire l'image Docker
build: train
	docker build -t $(IMAGE_NAME):$(VERSION) .
	@echo "🛠️ Docker image built: $(IMAGE_NAME):$(VERSION)"

# Lancer le conteneur Docker
run_docker:
	@echo "🔍 Checking for existing container..."
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@echo "🌐 Starting new container..."
	docker run -d --rm --name $(CONTAINER_NAME) -p 8000:8000 $(IMAGE_NAME):$(VERSION)
	@echo "🌐 Docker container running on port 8000."

# Pousser l'image sur Docker Hub
push: build
	@echo "🔍 Logging into Docker Hub..."
	@if [ -z "$(DOCKER_HUB_PASSWORD)" ]; then echo "❌ DOCKER_HUB_PASSWORD not set."; exit 1; fi
	docker login -u $(DOCKER_HUB_USER) -p $(DOCKER_HUB_PASSWORD) || { echo "❌ Docker login failed."; exit 1; }
	@echo "🔍 Tagging image for Docker Hub..."
	docker tag $(IMAGE_NAME):$(VERSION) $(DOCKER_HUB_USER)/$(IMAGE_NAME):$(VERSION)
	@echo "🔍 Pushing image to Docker Hub..."
	docker push $(DOCKER_HUB_USER)/$(IMAGE_NAME):$(VERSION) || { echo "❌ Push failed."; exit 1; }
	@echo "📤 Docker image pushed to $(DOCKER_HUB_USER)/$(IMAGE_NAME):$(VERSION)"

# Nettoyer les images et conteneurs Docker
docker_clean:
	docker rm -f $(CONTAINER_NAME) || true
	docker rmi -f $(IMAGE_NAME):$(VERSION) || true
	docker-compose down
	@echo "🧹 Cleaned Docker images and containers."

docker:
	docker-compose up -d

all_docker: build run_docker push
	@echo "🔄 Full Docker pipeline executed."

all: install format lint docker prepare train evaluate build push all_docker
	@echo "🔄 Full pipeline executed."