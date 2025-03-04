pipeline {
    agent any
    stages {
        stage('Récupérer le code') {
            steps {
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: '*/main']],
                    doGenerateSubmoduleConfigurations: false,
                    extensions: [
                        [$class: 'CloneOption', depth: 0, noTags: false, reference: '', shallow: false],
                        [$class: 'GitLFSPull'],
                        [$class: 'CheckoutOption', timeout: 30]
                    ],
                    userRemoteConfigs: [[url: 'https://github.com/MeriemMojaat/mlops.git']],
                    gitTool: 'Default'
                ])
                sh 'git config http.postBuffer 524288000'
            }
        }
        stage('Installer les dépendances') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }
        stage('Vérifier le code') {
            steps {
                sh '''
                    . venv/bin/activate
                    make lint
                '''
            }
        }
        stage('Vérifier Elasticsearch et Kibana') {
            steps {
                sh 'curl -u elastic:changeme http://172.27.70.134:9200 || echo "Elasticsearch not running"'
                sh 'curl http://172.27.70.134:5601 || echo "Kibana not running"'
            }
        }
        stage('Vérifier FastAPI') {
            steps {
                sh 'curl http://172.27.70.134:8000 || echo "FastAPI not running"'
            }
        }
        stage('Lancer Flask') {
            steps {
                sh 'pkill -f "python app1.py" || true'
                sh '''
                    . venv/bin/activate
                    make run_flask &
                '''
                sh 'sleep 10'
                sh 'curl http://172.27.70.134:5000'
            }
        }
        stage('Réentraîner le modèle via Flask') {
            steps {
                sh 'curl -X POST http://172.27.70.134:5000/retrain -H "Content-Type: application/json" -d \'{"learning_rate": 0.1, "n_estimators": 100, "max_depth": 6, "min_child_weight": 1, "gamma": 0, "subsample": 0.8, "colsample_bytree": 0.8}\''
            }
        }
        stage('Tester FastAPI pour prédiction') {
            steps {
                sh 'curl -X POST http://172.27.70.134:8000/predict -H "Content-Type: application/json" -d \'{"State": "CA", "Account_length": 50, "Area_code": 415, "International_plan": "No", "Voice_mail_plan": "Yes", "Number_vmail_messages": 10, "Total_day_minutes": 200.5, "Total_day_calls": 100, "Total_day_charge": 34.08, "Total_eve_minutes": 180.3, "Total_eve_calls": 90, "Total_eve_charge": 15.33, "Total_night_minutes": 150.2, "Total_night_calls": 80, "Total_night_charge": 6.76, "Total_intl_minutes": 10.5, "Total_intl_calls": 5, "Total_intl_charge": 2.84, "Customer_service_calls": 2}\' || echo "FastAPI prediction failed"'
            }
        }
    }
    post {
        always {
            node('') {
                sh 'pkill -f "python app1.py" || true'
            }
        }
    }
}