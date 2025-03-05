import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Percorsi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)

class QuantumAttentionLayer(nn.Module):
    def __init__(self, input_size, num_qubits=4):
        super().__init__()
        self.num_qubits = num_qubits
        self.simulator = AerSimulator()
        self.rotation_weights = nn.Parameter(torch.randn(num_qubits, 3, dtype=torch.float32))

    def quantum_circuit(self, input_vector):
        qc = QuantumCircuit(self.num_qubits)
        for i in range(min(self.num_qubits, len(input_vector))):
            theta_y = (input_vector[i].item() * self.rotation_weights[i, 0]).item()
            theta_z = (input_vector[i].item() * self.rotation_weights[i, 1]).item()
            qc.ry(theta_y, i)
            qc.rz(theta_z, i)
        qc.cx(0, 1)
        qc.measure_all()
        return qc

    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        outputs = torch.zeros_like(x, dtype=torch.float32)
        qc = self.quantum_circuit(x[0, 0])
        transpiled_qc = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled_qc, shots=32)
        result = job.result()
        counts = result.get_counts()
        state_vector = [int(state, 2) for state in counts.keys()]
        output_value = np.mean(state_vector)
        outputs.fill_(output_value)
        return outputs

class QuantumTransformer(nn.Module):
    def __init__(self, input_size, num_layers=40, num_qubits=2):
        super().__init__()
        self.quantum_layers = nn.ModuleList([
            QuantumAttentionLayer(input_size, num_qubits)
        ])
        self.output_layer = nn.Linear(input_size, input_size, dtype=torch.float32)

    def forward(self, x):
        for layer in self.quantum_layers:
            x = layer(x)
        return self.output_layer(x)

class DatasetManager:
    @staticmethod
    def load_documents(directory=DATASET_DIR):
        documents = []
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                    documents.extend(f.readlines())
        return ["io sono " + doc.strip() for doc in documents][:5]

def train_model(model, X):
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for epoch in range(12):
        optimizer.zero_grad()
        predictions = model(X)
        y_train = torch.tanh(X)
        loss = loss_fn(predictions, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def chat(model, vectorizer):
    print("Inizia a chattare (scrivi 'esci' per uscire):")
    while True:
        text = input("Tu: ")
        if text.lower() == "esci":
            break
        X = vectorizer.transform([text]).toarray()
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
        output = model(X)
        print(f"Modello: {output.mean().item()}")

def main():
    # Dataset
    documents = DatasetManager.load_documents()
    if not documents:
        print("⚠️ Nessun documento trovato!")
        return

    # Vettorizzazione
    vectorizer = CountVectorizer(max_features=4, binary=True)
    X = vectorizer.fit_transform(documents).toarray()
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

    print(f"Forma di X: {X.shape}")

    # Modello e training
    model = QuantumTransformer(input_size=X.shape[2])
    train_model(model, X)

    # Top features
    predictions = model(X)
    top_features_indices = torch.topk(predictions.mean(dim=1), k=2).indices.numpy()
    top_features = vectorizer.get_feature_names_out()[top_features_indices.flatten()].tolist()

    print("🚀 Training completato!")
    print(f"Documenti: {len(documents)}")
    print(f"Top Features: {top_features}")

    # Chat
    chat(model, vectorizer)

if __name__ == "__main__":
    main()