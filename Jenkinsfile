pipeline {
    agent any
    environment {
        DOCKER_HUB_CREDENTIALS = credentials('docker-hub-credentials')
    }
    stages {
        stage('Checkout') {
            steps {
                git url: 'https://github.com/meriemmojaat/ml_project.git', branch: 'main'
            }
        }
        stage('Install Dependencies') {
            steps {
                sh 'make install'
            }
        }
        stage('Lint Code') {
            steps {
                sh 'make lint'
            }
        }
        stage('Format Code') {
            steps {
                sh 'make format'
            }
        }
        stage('Prepare Data') {
            steps {
                sh 'make prepare'
            }
        }
        stage('Train and Save Model') {
            steps {
                sh 'make train'
            }
        }
        stage('Evaluate Model') {
            steps {
                sh 'make evaluate'
            }
        }
        stage('Start Docker Compose') {
            steps {
                sh 'make docker'
            }
        }
        stage('Build Docker Image') {
            steps {
                sh 'make build'
            }
        }
        stage('Run Docker Container') {
            steps {
                sh 'make run_docker'
            }
        }
        stage('Push to Docker Hub') {
            steps {
                sh 'make push DOCKER_HUB_PASSWORD=$DOCKER_HUB_CREDENTIALS_PSW'
            }
        }
        stage('Clean Up') {
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
