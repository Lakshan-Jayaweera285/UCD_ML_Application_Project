from pymongo import MongoClient
from datetime import datetime

DB_NAME = "cifar10_metrics_db"
COLLECTION_NAME = "metrics"
COLLECTION_TRAINING_DETAILS = "training_details"
client = MongoClient('mongodb://localhost:27017/')  
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
training_details_collection = db[COLLECTION_TRAINING_DETAILS]

def init_db():
    pass

def add_metric(accuracy: float, f1_score: float, precision: float, recall: float):
    metric = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": accuracy,
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall
    }
    collection.insert_one(metric)

#fetch evaluated metrics from MongoDB
def get_metrics(n: int = None):
    query = {}
    sort_order = [("timestamp", -1)]  
    
    if n:
        metrics = collection.find(query).sort(sort_order).limit(n)
    else:
        metrics = collection.find(query).sort(sort_order)
    
    return [{"timestamp": metric["timestamp"], 
             "accuracy": metric["accuracy"], 
             "f1_score": metric["f1_score"], 
             "precision": metric["precision"], 
             "recall": metric["recall"]} for metric in metrics]

def save_training_details(epochs, batch_size, learning_rate, validation_split, test_acc, test_loss, val_acc, val_loss):
    training_detail = {
        "timestamp": datetime.now().isoformat(),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_split": validation_split,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "validation_accuracy": val_acc,
        "validation_loss": val_loss
    }
    training_details_collection.insert_one(training_detail)
    print("Training details saved to MongoDB")

def get_training_details(n: int = None):
    query = {}
    sort_order = [("timestamp", -1)] 
    
    try:
        if n:
            details = training_details_collection.find(query).sort(sort_order).limit(n)
        else:
            details = training_details_collection.find(query).sort(sort_order)
        return [
            {
                "timestamp": detail["timestamp"],
                "epochs": detail["epochs"],
                "batch_size": detail["batch_size"],
                "learning_rate": detail["learning_rate"],
                "validation_split": detail["validation_split"],
                "test_accuracy": detail["test_accuracy"],
                "test_loss": detail["test_loss"],
                "validation_accuracy": detail["validation_accuracy"],
                "validation_loss": detail["validation_loss"],
            }
            for detail in details
        ]
    except Exception as e:
        print(f"Error retrieving training details: {e}")
        return []
