import numpy as np
import pandas as pd
from collections import Counter
import itertools
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import skfuzzy as fuzz
import cv2
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



class KMeansCustom:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels_ = None

    def fit(self, X):
        n_samples, n_features = X.shape
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            old_labels = self.labels_ if self.labels_ is not None else None
            self.labels_ = self._assign_clusters(X)

            for k in range(self.n_clusters):
                if np.sum(self.labels_ == k) > 0:
                    self.centroids[k] = np.mean(X[self.labels_ == k], axis=0)

            if old_labels is not None and np.all(old_labels == self.labels_):
                break

        return self

    def _assign_clusters(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign_clusters(X)


class FCMCustom:
    def __init__(self, n_clusters=3, max_iters=100, m=2.0):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.m = m
        self.centroids = None
        self.u = None

    def fit(self, X):
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X.T, self.n_clusters, self.m, error=0.005, maxiter=self.max_iters, init=None
        )
        self.centroids = cntr
        self.u = u
        return self

    def predict(self, X):
        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
            X.T, self.centroids, self.m, error=0.005, maxiter=self.max_iters
        )
        return np.argmax(u, axis=0)


class ClusteringMetrics:
    @staticmethod
    def f1_score(true_labels, pred_labels):
        def get_pairs(labels):
            n = len(labels)
            pairs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    if labels[i] == labels[j]:
                        pairs.add((min(i, j), max(i, j)))
            return pairs

        true_pairs = get_pairs(true_labels)
        pred_pairs = get_pairs(pred_labels)

        tp = len(true_pairs & pred_pairs)
        fp = len(pred_pairs - true_pairs)
        fn = len(true_pairs - pred_pairs)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    @staticmethod
    def rand_index(true_labels, pred_labels):
        n = len(true_labels)
        a = b = c = d = 0

        for i, j in itertools.combinations(range(n), 2):
            same_true = true_labels[i] == true_labels[j]
            same_pred = pred_labels[i] == pred_labels[j]

            if same_true and same_pred:
                a += 1
            elif same_true and not same_pred:
                b += 1
            elif not same_true and same_pred:
                c += 1
            else:
                d += 1

        return (a + d) / (a + b + c + d) if (a + b + c + d) > 0 else 0


class ImageProcessor:
    @staticmethod
    def load_and_preprocess_image(file_path):
        # Đọc ảnh
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("Không thể đọc ảnh")

        # Chuyển đổi sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize ảnh nếu quá lớn
        max_dimension = 500
        height, width = image_rgb.shape[:2]
        if height > max_dimension or width > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))

        # Chuẩn bị dữ liệu cho clustering
        pixels = image_rgb.reshape((-1, 3))
        return image_rgb, pixels

    @staticmethod
    def reshape_predictions(predictions, original_shape):
        return predictions.reshape(original_shape[:2])


class ClusteringApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ứng dụng Phân cụm")
        self.geometry("1200x800")

        # Khởi tạo biến
        self.X = None
        self.y = None
        self.image = None
        self.image_pixels = None
        self.current_results = None

        self.create_widgets()

    def create_widgets(self):
        # Frame chính
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame điều khiển
        control_frame = ttk.LabelFrame(main_frame, text="Điều khiển")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Chọn loại dữ liệu
        self.data_type_var = tk.StringVar(value="iris")
        ttk.Radiobutton(control_frame, text="IRIS Dataset",
                        variable=self.data_type_var, value="iris").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(control_frame, text="Ảnh giao thông",
                        variable=self.data_type_var, value="image").pack(side=tk.LEFT, padx=5)

        # Chọn thuật toán
        self.algorithm_var = tk.StringVar(value="KMeans")
        ttk.Radiobutton(control_frame, text="K-Means",
                        variable=self.algorithm_var, value="KMeans").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(control_frame, text="FCM",
                        variable=self.algorithm_var, value="FCM").pack(side=tk.LEFT, padx=5)

        # Số cụm
        ttk.Label(control_frame, text="Số cụm:").pack(side=tk.LEFT, padx=5)
        self.n_clusters_var = tk.StringVar(value="3")
        ttk.Entry(control_frame, textvariable=self.n_clusters_var, width=5).pack(side=tk.LEFT, padx=5)

        # Nút tải dữ liệu
        ttk.Button(control_frame, text="Tải dữ liệu",
                   command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Thực hiện phân cụm",
                   command=self.train_model).pack(side=tk.LEFT, padx=5)

        # Frame hiển thị
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Frame kết quả bên trái
        left_frame = ttk.LabelFrame(display_frame, text="Kết quả đánh giá")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Text hiển thị metrics
        self.metrics_text = tk.Text(left_frame, height=10)
        self.metrics_text.pack(fill=tk.X, padx=5, pady=5)

        # Biểu đồ metrics
        self.fig_metrics = Figure(figsize=(5, 4))
        self.canvas_metrics = FigureCanvasTkAgg(self.fig_metrics, master=left_frame)
        self.canvas_metrics.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Frame kết quả bên phải
        right_frame = ttk.LabelFrame(display_frame, text="Kết quả phân cụm")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Biểu đồ phân cụm
        self.fig_cluster = Figure(figsize=(5, 4))
        self.canvas_cluster = FigureCanvasTkAgg(self.fig_cluster, master=right_frame)
        self.canvas_cluster.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_data(self):
        data_type = self.data_type_var.get()
        if data_type == "iris":
            self.load_iris_data()
        else:
            self.load_image_data()

    def load_iris_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Data files", "*.data"), ("All files", "*.*")]
        )
        if file_path:
            try:
                # Đọc dữ liệu IRIS
                column_names = ['sepal_length', 'sepal_width',
                                'petal_length', 'petal_width', 'class']
                data = pd.read_csv(file_path, header=None, names=column_names)

                # Chuẩn hóa dữ liệu
                scaler = StandardScaler()
                self.X = scaler.fit_transform(data.iloc[:, :-1].values)
                self.y = pd.factorize(data['class'])[0]

                self.image = None
                self.image_pixels = None

                self.metrics_text.delete(1.0, tk.END)
                self.metrics_text.insert(tk.END, "Đã tải dữ liệu IRIS thành công!\n")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi tải dữ liệu IRIS: {str(e)}")

    def load_image_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.image, self.image_pixels = ImageProcessor.load_and_preprocess_image(file_path)
                self.X = self.image_pixels
                self.y = None

                self.metrics_text.delete(1.0, tk.END)
                self.metrics_text.insert(tk.END, "Đã tải ảnh thành công!\n")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi tải ảnh: {str(e)}")

    def train_model(self):
        if self.X is None:
            messagebox.showerror("Lỗi", "Vui lòng tải dữ liệu trước!")
            return

        try:
            # Lấy các tham số
            n_clusters = int(self.n_clusters_var.get())
            use_fcm = self.algorithm_var.get() == "FCM"
            data_type = self.data_type_var.get()

            # Thực hiện phân cụm
            if use_fcm:
                model = FCMCustom(n_clusters=n_clusters)
            else:
                model = KMeansCustom(n_clusters=n_clusters)

            model.fit(self.X)
            pred_labels = model.predict(self.X)

            # Tính toán metrics cho dữ liệu IRIS
            if data_type == "iris" and self.y is not None:
                f1 = ClusteringMetrics.f1_score(self.y, pred_labels)
                rand = ClusteringMetrics.rand_index(self.y, pred_labels)

                self.metrics_text.insert(tk.END, f"\nF1-Score: {f1:.4f}\n")
                self.metrics_text.insert(tk.END, f"RAND Index: {rand:.4f}\n")

                self.update_metrics_chart(f1, rand)

            # Lưu kết quả
            self.current_results = {
                'predicted_labels': pred_labels,
                'centroids': model.centroids
            }

            # Cập nhật visualization
            self.update_clustering_visualization(data_type)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi phân cụm: {str(e)}")

    def update_metrics_chart(self, f1_score, rand_index):
        self.fig_metrics.clear()
        ax = self.fig_metrics.add_subplot(111)

        metrics = ['F1-Score', 'RAND Index']
        values = [f1_score, rand_index]
        colors = ['#2ecc71', '#3498db']

        bars = ax.bar(metrics, values, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Giá trị')
        ax.set_title('Đánh giá Phân cụm')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom')

        self.canvas_metrics.draw()

    def update_clustering_visualization(self, data_type):
        self.fig_cluster.clear()
        ax = self.fig_cluster.add_subplot(111)

        if data_type == "iris":
            # Visualization cho dữ liệu IRIS
            scatter = ax.scatter(self.X[:, 0], self.X[:, 1],
                                 c=self.current_results['predicted_labels'],
                                 cmap='viridis')

            # Vẽ centroids
            ax.scatter(self.current_results['centroids'][:, 0],
                       self.current_results['centroids'][:, 1],
                       c='red', marker='x', s=200, linewidths=3,
                       label='Centroids')

            ax.set_title('Phân cụm dữ liệu IRIS')
            ax.set_xlabel('Đặc trưng 1')
            ax.set_ylabel('Đặc trưng 2')
            ax.legend()

        else:
            # Visualization cho ảnh
            if self.image is not None:
                # Reshape predicted labels về kích thước ảnh gốc
                segmented_image = np.zeros_like(self.image)
                labels_reshaped = self.current_results['predicted_labels'].reshape(self.image.shape[:2])

                # Tạo màu cho từng cụm
                n_clusters = len(np.unique(self.current_results['predicted_labels']))
                colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))[:, :3]

                # Gán màu cho từng pixel theo nhãn cụm
                for i in range(n_clusters):
                    mask = labels_reshaped == i
                    segmented_image[mask] = colors[i] * 255

                # Hiển thị ảnh gốc và ảnh đã phân cụm
                ax.imshow(segmented_image.astype(np.uint8))
                ax.set_title('Kết quả phân cụm ảnh')
                ax.axis('off')

        self.canvas_cluster.draw()

    def run(self):
        self.mainloop()


def main():
    app = ClusteringApp()
    app.run()


if __name__ == "__main__":
    main()