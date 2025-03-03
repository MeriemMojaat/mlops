pipeline {
    agent any
    environment {
        DOCKER_HUB_CREDENTIALS = credentials('docker-hub-credentials')
    }
    stages {
        stage('Récupérer le code') {
            steps {
                git url: 'https://github.com/MeriemMojaat/mlops.git', branch: 'main'
            }
        }
        stage('Démarrer Elasticsearch et Kibana') {
            steps {
                sh 'make docker'
            }
        }
        stage('Installer les dépendances') {
            steps {
                sh 'make install'
            }
        }
        stage('Vérifier le code') {
            steps {
                sh 'make lint'
            }
        }
        stage('Formatter le code') {
            steps {
                sh 'make format'
            }
        }
        stage('Préparer les données') {
            steps {
                sh 'make prepare'
            }
        }
        stage('Entraîner et sauvegarder le modèle') {
            steps {
                sh 'make train'
            }
        }
        stage('Évaluer le modèle') {
            steps {
                sh 'make evaluate'
            }
        }
        stage('Construire l’image Docker') {
            steps {
                sh 'make build'
            }
        }
        stage('Lancer le conteneur Docker') {
            steps {
                sh 'make run_docker'
            }
        }
        stage('Pousser vers Docker Hub') {
            steps {
                sh 'make push DOCKER_HUB_PASSWORD=$DOCKER_HUB_CREDENTIALS_PSW'
            }
        }
        stage('Nettoyer') {
            steps {
                sh 'make docker_clean'
            }
        }
    }
    post {
        always {
            sh 'docker logout'
            archiveArtifacts artifacts: 'traces.txt, *.json, loss_plot.png', allowEmptyArchive: true
        }
    }
}
