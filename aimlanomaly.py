import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler , LabelEncoder
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog , messagebox

df = None

def upload_csv():
    global df

    file_path = filedialog.askopenfilename(
        title = "Selects a csv file",
        filetypes = [("CSV Files","*.csv")]
    )

    if file_path:
        try:
            df = pd.read_csv(file_path)
            messagebox.showinfo("Succes! ","CSV file uploaded successfully")
            print(df.head(5))
        except Exception as e:
            messagebox.showerror("Failed to load CSV file")

def visualize_original_dataset():
    global df

    if df is None:
        messagebox.showwarning("Warning", "Please upload a CSV file first!")
        return
    
    df = df.copy()
    df = df.drop_duplicates()

    for col in df.columns:
        if 'id' in col.lower() or 'name' in col.lower():
            df.drop(columns=[col], inplace=True)

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    scaler = RobustScaler()      
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])               
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(8, 5))
        plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.5)
        plt.title("Preprocessed Data")
        plt.xlabel(numeric_cols[0])
        plt.ylabel(numeric_cols[1])
        plt.show()
    return df

def detect_anomalies():
    global df

    if df is None:
        messagebox.showwarning("Warning", "Please upload a CSV file first!")
        return
    preprocesed_df = visualize_original_dataset()
    model = IsolationForest(contamination=0.0017,random_state=42)
    predictions = model.fit_predict(preprocesed_df)

    preprocesed_df['Anomaly'] = predictions

    anomaly_count = np.sum(predictions == -1)
    messagebox.showinfo("Detection completed ! Number of anamolies found",anomaly_count)

    normal_data = preprocesed_df[preprocesed_df['Anomaly'] == 1]
    anomaly_data = preprocesed_df[preprocesed_df['Anomaly'] == -1]

    if preprocesed_df.shape[1] >= 2:
        cols = preprocesed_df.columns[:2]
        
    plt.figure(figsize=(8, 5))
    plt.scatter(normal_data[cols[0]], normal_data[cols[1]],color='#22d3ee', label="Normal")
    plt.scatter(anomaly_data[cols[0]], anomaly_data[cols[1]], color='#f43f5e', label="Anomaly")
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.title("Anomaly Detection using Isolation Forest")
    plt.legend()
    plt.show()

# root = tk.Tk()
# root.title("CSV Anomaly Detection")
# root.geometry("400x250")

# upload_btn = tk.Button(root, text="Upload CSV File", command=upload_csv, font=("Arial", 12),fg="#f8fafc",bg='#334155')
# upload_btn.pack(pady=20)

# detect_btn = tk.Button(root, text="Detect Anomalies", command=detect_anomalies, font=("Arial", 12),fg="#f8fafc",bg='#334155')
# detect_btn.pack(pady=20)

# visualize_btn = tk.Button(root, text="Visualize Original Data", command=visualize_original_dataset, font=("Arial", 12),fg="#f8fafc",bg='#334155')
# visualize_btn.pack(pady=20)

# root.configure(bg='#0f172a')
# root.mainloop()


root = tk.Tk()
root.title("AI Anomaly Detector")
root.state('zoomed') 
root.configure(bg="#0f172a")

title = tk.Label(root, text="AI Anomaly Detection System",
                 font=("Segoe UI", 18, "bold"),
                 fg="#f8fafc", bg="#0f172a")
title.pack(pady=(20, 5))

subtitle = tk.Label(root, text="Upload CSV • Visualize • Detect Outliers",
                    font=("Segoe UI", 10),
                    fg="#94a3b8", bg="#0f172a")
subtitle.pack(pady=(0, 20))

card = tk.Frame(root, bg="#1e293b", bd=0)
card.pack(padx=20, pady=10, fill="both", expand=True)

status_label = tk.Label(card, text="Status: Waiting for input...",
                        font=("Segoe UI", 10),
                        fg="#cbd5f5", bg="#1e293b")
status_label.pack(pady=10)


def create_button(text, command):
    btn = tk.Button(card,
                    text=text,
                    command=command,
                    font=("Segoe UI", 11, "bold"),
                    fg="#f8fafc",
                    bg="#334155",
                    activebackground="#475569",
                    activeforeground="white",
                    bd=0,
                    padx=10,
                    pady=12,
                    cursor="hand2")

    
    def on_enter(e):
        btn['bg'] = "#475569"
    def on_leave(e):
        btn['bg'] = "#334155"

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

    return btn


def upload_action():
    upload_csv()
    status_label.config(text="Status: CSV Uploaded Successfully")

def detect_action():
    detect_anomalies()
    status_label.config(text="Status: Anomaly Detection Completed")

def visualize_action():
    visualize_original_dataset()
    status_label.config(text="Status: Data Visualized")

btn_upload = create_button("Upload CSV File", upload_action)
btn_upload.pack(pady=10, fill="x", padx=40)

btn_visualize = create_button("Visualize Data", visualize_action)
btn_visualize.pack(pady=10, fill="x", padx=40)

btn_detect = create_button("Detect Anomalies", detect_action)
btn_detect.pack(pady=10, fill="x", padx=40)


footer = tk.Label(root,
                  text="Built with Isolation Forest | AIML Project",
                  font=("Segoe UI", 8),
                  fg="#64748b",
                  bg="#0f172a")
footer.pack(side="bottom", pady=10)

root.mainloop()

