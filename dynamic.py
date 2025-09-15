import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
import os
from collections import defaultdict

class PerUserKeystrokeAuth:
    def __init__(self, data_path, sequence_length=50, min_samples_per_user=100):
        """Initialize the Per-User Keystroke Authentication System"""
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.min_samples_per_user = min_samples_per_user
        self.user_models = {}
        self.user_scalers = {}
        self.timing_columns = ['DU.key1.key1', 'DD.key1.key2', 'DU.key1.key2',
                             'UD.key1.key2', 'UU.key1.key2 ']  # Note the space after UU.key1.key2
        self.user_stats = {}
        
    def load_data(self):
        """Load and preprocess the keystroke data"""
        print("Loading data from:", self.data_path)
        df = pd.read_csv(self.data_path, low_memory=False)
        
        # Convert timing columns to numeric, handling any spaces
        for col in self.timing_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values with column means
        df = df.fillna(df[self.timing_columns].mean())
        
        # Filter users with sufficient data
        user_counts = df['participant'].value_counts()
        valid_users = user_counts[user_counts >= self.min_samples_per_user].index
        df_filtered = df[df['participant'].isin(valid_users)]
        
        print(f"Found {len(valid_users)} users with at least {self.min_samples_per_user} samples")
        print(f"Users: {list(valid_users)}")
        
        return df_filtered
    
    def create_user_sequences(self, df, target_user):
        """Create sequences for a specific user (genuine vs impostor)"""
        sequences = []
        labels = []
        
        # Get genuine user data
        user_data = df[df['participant'] == target_user][self.timing_columns].values
        
        # Create genuine sequences
        for i in range(0, len(user_data) - self.sequence_length + 1):
            seq = user_data[i:i + self.sequence_length]
            sequences.append(seq)
            labels.append(1)  # Genuine user
        
        # Get impostor data (all other users)
        impostor_data = df[df['participant'] != target_user][self.timing_columns].values
        
        # Create impostor sequences (limit to balance the dataset)
        num_genuine = len(sequences)
        impostor_sequences_needed = min(num_genuine, len(impostor_data) - self.sequence_length + 1)
        
        # Randomly sample impostor sequences
        if impostor_sequences_needed > 0:
            impostor_indices = np.random.choice(
                len(impostor_data) - self.sequence_length + 1, 
                size=impostor_sequences_needed, 
                replace=False
            )
            
            for idx in impostor_indices:
                seq = impostor_data[idx:idx + self.sequence_length]
                sequences.append(seq)
                labels.append(0)  # Impostor
        
        return np.array(sequences), np.array(labels)
    
    def build_model(self):
        """Build the CNN-LSTM model for a single user"""
        input_layer = Input(shape=(self.sequence_length, len(self.timing_columns)))
        
        # CNN layers
        conv1 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(input_layer)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        conv2 = Conv1D(32, kernel_size=3, activation='relu', padding='same')(pool1)
        pool2 = MaxPooling1D(pool_size=2)(conv2)
        
        # LSTM layers
        lstm1 = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(pool2)
        lstm2 = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(lstm1)
        
        # Dense layers
        dense1 = Dense(64, activation='relu')(lstm2)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(32, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        output = Dense(1, activation='sigmoid')(dropout2)
        
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_user_model(self, user, df, epochs=50, batch_size=32, validation_split=0.2):
        """Train a model for a specific user"""
        print(f"\nTraining model for user: {user}")
        
        try:
            # Create sequences for this user
            X, y = self.create_user_sequences(df, user)
            print(f"Created {len(X)} sequences ({np.sum(y)} genuine, {len(y) - np.sum(y)} impostor)")
            
            # Initialize scaler for this user
            scaler = StandardScaler()
            
            # Scale the features
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = scaler.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Build and train the model
            model = self.build_model()
            
            # Add callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate the model
            y_pred_prob = model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            
            # Calculate FAR and FRR
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Store model and scaler
            self.user_models[user] = model
            self.user_scalers[user] = scaler
            
            # Store statistics
            self.user_stats[user] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'far': far,
                'frr': frr,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            print(f"User {user} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, FAR: {far:.4f}, FRR: {frr:.4f}")
            
            return history
            
        except Exception as e:
            print(f"Error training model for user {user}: {str(e)}")
            return None
    
    def train_all_users(self, epochs=50, batch_size=32):
        """Train models for all users"""
        print("Starting training for all users...")
        
        # Load data
        df = self.load_data()
        
        # Get list of users
        users = df['participant'].unique()
        
        # Train model for each user
        histories = {}
        for user in users:
           history = self.train_user_model(user, df, epochs, batch_size)
        # Use below two lines if you want to train specific number of models.
        # For example replace n with 1 to train one model
        #user = users[n]  # Get the first user
        #history = self.train_user_model(user, df, epochs, batch_size)
        if history is not None:
            histories[user] = history
        
        # Save all models and scalers
        self.save_models()
        
        # Print summary statistics
        self.print_summary_stats()
        
        return histories
    
    def authenticate_user(self, user, test_data, threshold=0.5):
        """Authenticate a specific user with their personalized model"""
        if user not in self.user_models:
            #print(f"Error: No model found for user {user}")
            print(f"Warning: {user} is not authenticated")
            return None
        
        model = self.user_models[user]
        scaler = self.user_scalers[user]
        
        try:
            # Prepare test data
            if len(test_data) < self.sequence_length:
                print(f"Error: Need at least {self.sequence_length} samples for authentication")
                return None
            
            test_sequences = []
            for i in range(0, len(test_data) - self.sequence_length + 1):
                seq = test_data[i:i + self.sequence_length]
                test_sequences.append(seq)
            
            test_sequences = np.array(test_sequences)
            
            # Scale the test data
            test_reshaped = test_sequences.reshape(-1, test_sequences.shape[-1])
            test_scaled = scaler.transform(test_reshaped)
            test_scaled = test_scaled.reshape(test_sequences.shape)
            
            # Get predictions
            predictions = model.predict(test_scaled, verbose=0)
            
            # Use median instead of mean for more robust authentication
            median_score = np.median(predictions)
            mean_score = np.mean(predictions)
            
            # More conservative authentication - require multiple sequences to agree
            high_confidence_predictions = predictions[predictions > 0.7]
            low_confidence_predictions = predictions[predictions < 0.3]
            
            # Authentication decision
            authenticated = False
            confidence = float(median_score)
            
            if median_score > threshold:
                # Additional check: majority of predictions should be confident
                if len(high_confidence_predictions) > len(predictions) * 0.3:
                    authenticated = True
            
            return {
                'user': user,
                'authenticated': authenticated,
                'confidence': confidence,
                'mean_confidence': float(mean_score),
                'median_confidence': float(median_score),
                'threshold': threshold,
                'num_sequences': len(test_sequences),
                'high_confidence_ratio': len(high_confidence_predictions) / len(predictions)
            }
            
        except Exception as e:
            print(f"Error during authentication for user {user}: {str(e)}")
            return None
    
    def save_models(self):
        """Save all user models and scalers"""
        models_dir = "user_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        for user in self.user_models:
            model_path = os.path.join(models_dir, f"model_{user}.h5")
            scaler_path = os.path.join(models_dir, f"scaler_{user}.pkl")
            
            self.user_models[user].save(model_path)
            joblib.dump(self.user_scalers[user], scaler_path)
        
        # Save user stats
        stats_path = os.path.join(models_dir, "user_stats.pkl")
        joblib.dump(self.user_stats, stats_path)
        
        print(f"Saved {len(self.user_models)} user models to {models_dir}/")
    
    def load_models(self):
        """Load all user models and scalers"""
        models_dir = "user_models"
        
        if not os.path.exists(models_dir):
            print(f"Error: Models directory {models_dir} not found")
            return False
        
        try:
            # Load user stats
            stats_path = os.path.join(models_dir, "user_stats.pkl")
            if os.path.exists(stats_path):
                self.user_stats = joblib.load(stats_path)
            
            # Load models and scalers
            for filename in os.listdir(models_dir):
                if filename.startswith("model_") and filename.endswith(".h5"):
                    user = filename.replace("model_", "").replace(".h5", "")
                    
                    model_path = os.path.join(models_dir, filename)
                    scaler_path = os.path.join(models_dir, f"scaler_{user}.pkl")
                    
                    if os.path.exists(scaler_path):
                        self.user_models[user] = tf.keras.models.load_model(model_path)
                        self.user_scalers[user] = joblib.load(scaler_path)
            
            print(f"Loaded {len(self.user_models)} user models")
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def print_summary_stats(self):
        """Print summary statistics for all users"""
        if not self.user_stats:
            print("No statistics available")
            return
        
        print("\n" + "="*80)
        print("TRAINING SUMMARY STATISTICS")
        print("="*80)
        
        # Calculate overall statistics
        accuracies = [stats['accuracy'] for stats in self.user_stats.values()]
        f1_scores = [stats['f1'] for stats in self.user_stats.values()]
        fars = [stats['far'] for stats in self.user_stats.values()]
        frrs = [stats['frr'] for stats in self.user_stats.values()]
        
        print(f"Number of users trained: {len(self.user_stats)}")
        print(f"Average accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Average F1-score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"Average FAR: {np.mean(fars):.4f} ± {np.std(fars):.4f}")
        print(f"Average FRR: {np.mean(frrs):.4f} ± {np.std(frrs):.4f}")
        
        print("\nPer-user performance:")
        print("-" * 80)
        print(f"{'User':<10} {'Accuracy':<10} {'F1':<8} {'FAR':<8} {'FRR':<8} {'Samples':<10}")
        print("-" * 80)
        
        for user, stats in self.user_stats.items():
            print(f"{user:<10} {stats['accuracy']:<10.4f} {stats['f1']:<8.4f} "
                  f"{stats['far']:<8.4f} {stats['frr']:<8.4f} {stats['training_samples']:<10}")
    
    def plot_user_performance(self):
        """Plot performance metrics for all users"""
        if not self.user_stats:
            print("No statistics available for plotting")
            return
        
        users = list(self.user_stats.keys())
        accuracies = [self.user_stats[user]['accuracy'] for user in users]
        f1_scores = [self.user_stats[user]['f1'] for user in users]
        fars = [self.user_stats[user]['far'] for user in users]
        frrs = [self.user_stats[user]['frr'] for user in users]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        ax1.bar(users, accuracies, color='skyblue')
        ax1.set_title('Accuracy per User')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # F1 Score
        ax2.bar(users, f1_scores, color='lightgreen')
        ax2.set_title('F1-Score per User')
        ax2.set_ylabel('F1-Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # False Acceptance Rate
        ax3.bar(users, fars, color='salmon')
        ax3.set_title('False Acceptance Rate per User')
        ax3.set_ylabel('FAR')
        ax3.tick_params(axis='x', rotation=45)
        
        # False Rejection Rate
        ax4.bar(users, frrs, color='orange')
        ax4.set_title('False Rejection Rate per User')
        ax4.set_ylabel('FRR')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def main():
    # Initialize the per-user authentication system
    auth_system = PerUserKeystrokeAuth(
        data_path='/Users/venugopal/Downloads/free-text.csv',
        sequence_length=50,
        min_samples_per_user=100
    )
    
    # Train models for all users
    print("Training personalized models for each user...")
    histories = auth_system.train_all_users(epochs=30, batch_size=32)
    
    # Plot performance metrics
    auth_system.plot_user_performance()
    
    # Example authentication
    print("\n" + "="*50)
    print("AUTHENTICATION EXAMPLE")
    print("="*50)
    
    # Load sample data for testing
    df = auth_system.load_data()
    if not df.empty:
        # Test with first user's data
        test_user = df['participant'].iloc[0]
        user_data = df[df['participant'] == test_user][auth_system.timing_columns].values
        
        # Take last 60 samples as test data
        test_data = user_data[-60:]
        
        # Authenticate
        result = auth_system.authenticate_user(test_user, test_data)
        if result:
            print(f"Authentication result for user {test_user}:")
            print(f"Authenticated: {result['authenticated']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Sequences analyzed: {result['num_sequences']}")

if __name__ == "__main__":
    main()