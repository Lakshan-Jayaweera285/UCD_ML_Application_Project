from pymongo import MongoClient
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def get_metrics_from_db():
    client = MongoClient("mongodb://localhost:27017/")  
    db = client["cifar10_metrics_db"]  
    collection = db["metrics"] 


    metrics = collection.find({}, {"_id": 0, "timestamp": 1, "accuracy": 1, "precision": 1, "recall": 1, "f1_score": 1}).sort("timestamp", -1) .limit(10)
    metrics_data = pd.DataFrame(list(metrics))

    client.close()  

    return metrics_data


def main():
    st.title("Model Evaluation Metrics")

    
    metrics_data = get_metrics_from_db()

    if not metrics_data.empty:

        metrics_data.reset_index(drop=True, inplace=True)
        metrics_data["Index"] = metrics_data.index + 1

        
        st.write("Metrics Data:", metrics_data)

        # Accuracy
        st.subheader("Accuracy Over Time")
        st.line_chart(metrics_data.set_index("timestamp")["accuracy"])

        st.subheader("Accuracy Over Time ")
        fig1, ax1 = plt.subplots()
        ax1.plot(11-metrics_data["Index"], metrics_data["accuracy"], color='blue', label="Accuracy")
        ax1.set_title("Accuracy Over Time")
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        st.pyplot(fig1)

        # Precision
        st.subheader("Precision Over Time")
        st.line_chart(metrics_data.set_index("timestamp")["precision"])

        st.subheader("Precision Over Time")
        fig2, ax2 = plt.subplots()
        ax2.plot(11-metrics_data["Index"], metrics_data["precision"], color='green', label="Precision")
        ax2.set_title("Precision Over Time")
        ax2.set_xlabel("Timestamp(Table ID)")
        ax2.set_ylabel("Precision")
        ax2.legend()
        st.pyplot(fig2)

        # Recall
        st.subheader("Recall Over Time")
        st.line_chart(metrics_data.set_index("timestamp")["recall"])

        st.subheader("Recall Over Time")
        fig3, ax3 = plt.subplots()
        ax3.plot(11-metrics_data["Index"], metrics_data["recall"], color='orange', label="Recall")
        ax3.set_title("Recall Over Time")
        ax3.set_xlabel("Timestamp(Table ID)")
        ax3.set_ylabel("Recall")
        ax3.legend()
        st.pyplot(fig3)

        # F1 Score
        st.subheader("F1 Score Over Time")
        st.line_chart(metrics_data.set_index("timestamp")["f1_score"])

        st.subheader("F1 Score Over Time")
        fig4, ax4 = plt.subplots()
        ax4.plot(11-metrics_data["Index"], metrics_data["f1_score"], color='red', label="F1 Score")
        ax4.set_title("F1 Score Over Time")
        ax4.set_xlabel("Timestamp(Table ID)")
        ax4.set_ylabel("F1 Score")
        ax4.legend()
        st.pyplot(fig4)

        # Combined graph two Accuracy and Precision metrics
        st.subheader("Accuracy and Precision")
        fig, ax = plt.subplots()
        ax.plot(11-metrics_data["Index"], metrics_data["accuracy"], label="Accuracy", color='red')
        ax.plot(11-metrics_data["Index"], metrics_data["precision"], label="Precision", color='green')
        

        ax.set_xlabel("Timestamp(Table ID)")
        ax.set_ylabel("Metric Value")
        ax.set_title("Evaluation Accuracy and Precision Over Time")
        ax.legend()

        
        st.pyplot(fig)

        # Combined graph two Recall and F1 Score
        st.subheader("Recall and F1 Score")
        fig5, ax5 = plt.subplots()
        ax5.plot(11-metrics_data["Index"], metrics_data["recall"], label="Recall", color='orange')
        ax5.plot(11-metrics_data["Index"], metrics_data["f1_score"], label="F1 Score", color='red')

        ax5.set_xlabel("Timestamp(Table ID)")
        ax5.set_ylabel("Metric Value")
        ax5.set_title("Evaluation Recall and f1 Score Over Time")
        ax5.legend()

        # Display 
        st.pyplot(fig5)

    else:
        st.write("No data found in the database")

if __name__ == "__main__":
    main()
