import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import warnings
import pickle
import os
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SARAH:
    def __init__(self):
        self.name = "SARAH"
        self.full_name = "Semi-Autonomous Response for Adaptive Helper"
        self.cnn_model = None
        self.vectorizer = None
        self.knowledge_base = {}
        self.conversation_history = []
        self.setup_directories()
        
        # Initialize components
        self.initialize_nlp()
        self.initialize_cnn()
        self.load_knowledge_base()
        
        # Enhanced responses
        self.responses = {
            'greeting': [
                "Hello! I'm SARAH, your adaptive helper.",
                "Greetings! SARAH systems online.",
                "Hi there! How can I assist you today?"
            ],
            'generic': [
                "I'm analyzing your request now.",
                "That's an interesting point.",
                "Let me process that information.",
                "My systems are evaluating your input.",
                "I have several responses to that."
            ],
            'image_query': [
                "I've analyzed the image and detected: {result}",
                "My visual systems identify: {result}",
                "Image classification complete: {result}"
            ],
            'learning': [
                "I've noted that information about '{topic}'.",
                "Adding '{topic}' to my knowledge base.",
                "I'll remember that about '{topic}'."
            ],
            'farewell': [
                "Goodbye! Shutting down systems.",
                "SARAH systems powering down.",
                "Until next time!"
            ]
        }
    
    def setup_directories(self):
        """Create necessary directories for data storage"""
        os.makedirs('data/knowledge', exist_ok=True)
        os.makedirs('data/models', exist_ok=True)
    
    def initialize_nlp(self):
        """Initialize NLP components"""
        # Simple TF-IDF vectorizer for text similarity
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Pre-load with some basic phrases
        sample_phrases = [
            "hello", "hi", "greetings",
            "what can you do", "your capabilities",
            "image", "picture", "recognize this",
            "tell me about", "what is", "who is",
            "goodbye", "exit", "quit"
        ]
        self.vectorizer.fit_transform(sample_phrases)
    
    def initialize_cnn(self):
        """Initialize or load CNN model"""
        model_path = 'data/models/cnn_model.h5'
        if os.path.exists(model_path):
            self.cnn_model = tf.keras.models.load_model(model_path)
            print(f"{self.name}: Loaded trained CNN model")
        else:
            self.cnn_model = self.build_cnn()
            print(f"{self.name}: Initialized new CNN model")
    
    def build_cnn(self):
        """Build CNN architecture"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def train_cnn(self):
        """Train CNN on MNIST dataset"""
        print(f"\n{self.name}: Preparing to train CNN on MNIST dataset...")
        
        # Load MNIST data
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        
        # Normalize pixel values
        train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
        test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
        
        print(f"{self.name}: Training CNN model...")
        self.cnn_model.fit(train_images, train_labels, epochs=5, batch_size=64)
        
        # Evaluate
        test_loss, test_acc = self.cnn_model.evaluate(test_images, test_labels)
        print(f"{self.name}: Test accuracy: {test_acc:.2f}")
        
        # Save model
        self.cnn_model.save('data/models/cnn_model.h5')
        print(f"{self.name}: CNN model trained and saved")
    
    def classify_image(self, image_array):
        """Classify an image using CNN"""
        if len(image_array.shape) == 2:
            image_array = np.expand_dims(image_array, axis=-1)
        image_array = np.expand_dims(image_array, axis=0)
        prediction = self.cnn_model.predict(image_array)
        return np.argmax(prediction)
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    
    def understand_query(self, query):
        """Basic NLP understanding of user query"""
        query = self.preprocess_text(query)
        query_vec = self.vectorizer.transform([query])
        
        # Compare with known phrases
        known_phrases = [
            "hello", "hi", "greetings",
            "what can you do", "your capabilities",
            "image", "picture", "recognize this",
            "tell me about", "what is", "who is",
            "goodbye", "exit", "quit"
        ]
        phrase_vecs = self.vectorizer.transform(known_phrases)
        similarities = cosine_similarity(query_vec, phrase_vecs)
        max_sim_idx = np.argmax(similarities)
        
        if similarities[0, max_sim_idx] > 0.5:
            phrase = known_phrases[max_sim_idx]
            if phrase in ["hello", "hi", "greetings"]:
                return "greeting"
            elif phrase in ["what can you do", "your capabilities"]:
                return "capabilities"
            elif phrase in ["image", "picture", "recognize this"]:
                return "image_query"
            elif phrase in ["tell me about", "what is", "who is"]:
                return "knowledge_query"
            elif phrase in ["goodbye", "exit", "quit"]:
                return "farewell"
        
        return "generic"
    
    def load_knowledge_base(self):
        """Load knowledge base from file"""
        kb_path = 'data/knowledge/knowledge_base.pkl'
        if os.path.exists(kb_path):
            with open(kb_path, 'rb') as f:
                self.knowledge_base = pickle.load(f)
            print(f"{self.name}: Loaded knowledge base with {len(self.knowledge_base)} entries")
    
    def save_knowledge_base(self):
        """Save knowledge base to file"""
        kb_path = 'data/knowledge/knowledge_base.pkl'
        with open(kb_path, 'wb') as f:
            pickle.dump(self.knowledge_base, f)
    
    def add_to_knowledge(self, topic, information):
        """Add information to knowledge base"""
        if topic not in self.knowledge_base:
            self.knowledge_base[topic] = []
        self.knowledge_base[topic].append({
            'information': information,
            'timestamp': datetime.now().isoformat()
        })
        self.save_knowledge_base()
        return random.choice(self.responses['learning']).format(topic=topic)
    
    def query_knowledge(self, topic):
        """Query knowledge base"""
        topic = topic.lower()
        if topic in self.knowledge_base:
            info = random.choice(self.knowledge_base[topic])
            return f"I know that {topic} is related to: {info['information']} (learned {info['timestamp']})"
        return f"I don't have information about {topic} yet. Would you like to teach me?"
    
    def respond(self, query):
        """Generate appropriate response based on query"""
        query_type = self.understand_query(query)
        
        # Log conversation
        self.conversation_history.append({
            'query': query,
            'type': query_type,
            'timestamp': datetime.now().isoformat()
        })
        
        if query_type == 'greeting':
            return random.choice(self.responses['greeting'])
        elif query_type == 'farewell':
            return random.choice(self.responses['farewell'])
        elif query_type == 'image_query':
            # For demo purposes, we'll simulate image input
            # In a real implementation, you'd load an actual image
            if not self.cnn_model:
                return "My image recognition system isn't ready yet."
            
            # Simulate a random digit image
            simulated_image = np.random.rand(28, 28)
            digit = self.classify_image(simulated_image)
            return random.choice(self.responses['image_query']).format(result=f"digit {digit}")
        elif query_type == 'knowledge_query':
            # Extract topic (very simple extraction for demo)
            topic = query.replace('?', '').split()[-1]
            return self.query_knowledge(topic)
        elif 'teach' in query.lower():
            # Simple learning mechanism
            try:
                parts = query.split('that', 1)[1].split('is', 1)
                topic = parts[0].strip()
                info = parts[1].strip()
                return self.add_to_knowledge(topic, info)
            except:
                return "I couldn't understand what you're trying to teach me. Format: 'teach that [topic] is [information]'"
        else:
            return random.choice(self.responses['generic'])
    
    def greet(self):
        """Display welcome message"""
        print(f"\nInitializing {self.name} ({self.full_name})")
        print("Systems ready. You can:")
        print("- Ask questions")
        print("- Teach me with 'teach that [topic] is [information]'")
        print("- Type 'exit' to quit\n")
    
    def run(self):
        """Main interaction loop"""
        self.greet()
        
        # Train CNN if not already trained
        if not os.path.exists('data/models/cnn_model.h5'):
            self.train_cnn()
        
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print(f"\n{self.name}: {random.choice(self.responses['farewell'])}\n")
                break
                
            response = self.respond(user_input)
            print(f"\n{self.name}: {response}\n")

if __name__ == "__main__":
    ai = SARAH()
    ai.run()