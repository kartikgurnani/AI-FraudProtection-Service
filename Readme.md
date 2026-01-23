🛡️ Fraud Shield – AI-Powered Fraud Detection System

Fraud Shield is an end-to-end AI-driven fraud detection system designed to identify and prevent fraudulent financial transactions in both batch and real-time environments.
The project combines Machine Learning, Deep Learning, and Real-Time Streaming to deliver a scalable, high-performance fraud detection pipeline.

This repository includes baseline ML models, advanced ensemble & deep learning models, and a real-time fraud monitoring system powered by Apache Kafka, along with a frontend dashboard and AI-powered chatbot integration.

🚀 Project Highlights

🔍 Dual-Model Strategy

Logistic Regression for interpretability & baseline comparison

Random Forest & Deep Neural Network (DNN) for high-accuracy detection

⚡ Real-Time Fraud Detection

Apache Kafka Producer–Consumer architecture for live transaction monitoring

🧠 Advanced ML Techniques

SMOTE for handling class imbalance

Feature importance & hyperparameter tuning

Cross-validation & ROC-AUC analysis

🤖 AI Chatbot Integration

Gemini AI-powered chatbot for fraud insights & user assistance

📊 Interactive Visualization

React.js dashboard displaying confusion matrices, metrics & reports

📈 Trained on 70,000+ Transactions

Optimized for precision, recall, F1-score & ROC-AUC

📂 Project Structure
AI-Fraud-Detection/
│── notebooks/        # Jupyter notebooks (EDA & model training)
│── client/           # React frontend dashboard
│── server/           # Backend (FastAPI/Flask) with ML/DL models
│── kafka/            # Kafka producer & consumer scripts
│── requirements.txt  # Python dependencies

📘 Notebooks Overview
📓 Notebook 1: Credit Card Fraud Detection (Logistic Regression)

A baseline model focused on interpretability and simplicity.

Key Steps

Data preprocessing & feature scaling

Exploratory Data Analysis (EDA)

Logistic Regression training

Evaluation using Accuracy, Precision, Recall, F1-Score

Confusion Matrix analysis

📓 Notebook 2: Synthetic Financial Fraud Detection (Random Forest)

A robust ensemble-based model for improved fraud detection.

Key Steps

Advanced preprocessing & feature importance analysis

Deeper EDA to uncover complex patterns

Random Forest training with hyperparameter tuning

Cross-validation & ROC-AUC evaluation

Performance comparison with Logistic Regression

🧠 Deep Learning Model

Architecture: Deep Neural Network (DNN)

Frameworks: TensorFlow & Keras

Optimization: SMOTE, dropout, batch normalization

Goal: High recall and precision for rare fraud cases

🛠️ Tech Stack
Backend & ML

Python, Pandas, NumPy

Scikit-learn, Imbalanced-learn

TensorFlow, Keras

FastAPI / Flask

Real-Time Streaming

Apache Kafka (Producer–Consumer model)

Frontend

React.js

JavaScript / TypeScript

AI & Tools

Gemini AI (Chatbot)

Matplotlib, Seaborn

⚙️ Installation & Setup
Prerequisites

Python 3.8+

Apache Kafka

Node.js & npm

Jupyter Notebook

1️⃣ Clone the Repository
git clone https://github.com/Prince200510/AI-Fraud-Detection.git
cd AI-Fraud-Detection

2️⃣ Install Python Dependencies
pip install -r requirements.txt

3️⃣ Start Apache Kafka
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka Broker
bin/kafka-server-start.sh config/server.properties

4️⃣ Run Backend Server
cd server
python app.py

5️⃣ Run Kafka Producer & Consumer
python producer.py   # Sends transactions
python consumer.py   # Detects fraud in real-time

6️⃣ Run Frontend Dashboard
cd client
npm install
npm start

📊 Model Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Confusion Matrix

🤝 Contributing

Contributions are welcome!
Feel free to fork the repository, open issues, or submit pull requests to enhance the system.

📜 License

This project is licensed under the MIT License.
See the LICENSE file for details.

👤 Author & Profile

GitHub:
🔗 https://github.com/kartikgurnani

LinkedIn:
🔗 https://in.linkedin.com/in/kartikgurnani

⭐ Acknowledgments

Special thanks to TensorFlow, Apache Kafka, Scikit-learn, OpenAI, and Gemini AI for providing the tools that made this project possible.