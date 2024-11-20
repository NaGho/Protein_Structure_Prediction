import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.preprocess import preprocess_data, encode_sequences
from models.LSTM import create_model
from tensorflow.keras.callbacks import EarlyStopping
from scripts.utils import plot_metrics

# Load the dataset
data_url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz"
print("Downloading dataset...")
protein_data = pd.read_csv(data_url, compression='gzip', sep="\t")

# Preprocess the dataset
processed_data = preprocess_data(protein_data)

# Encode sequences
X, y, tokenizer, label_encoder = encode_sequences(processed_data)

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = create_model(input_length=X.shape[1], num_classes=y.shape[1])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)

# Plot metrics
plot_metrics(history)

# Save model and encoders
model.save("protein_model.h5")
