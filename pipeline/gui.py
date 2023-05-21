import tkinter as tk
from tkinter import filedialog
import pandas as pd
import threading
from tkinter import ttk
import os
import shutil
from pathlib import Path

from interface import load_neural_network, inference_neural_network, train_neural_network

# Создание основного окна
window = tk.Tk()
window.title("Нейросеть для классификации лебедей")
window.geometry("600x400")

# Создание таблицы с результатами
pd_frame = tk.Frame(window)
pd_frame.pack(pady=10)

pd_table = ttk.Treeview(pd_frame)
pd_table['columns'] = ('Photo', 'Class', 'Count', 'Confidence')
pd_table.column('#0', width=0, stretch='no')
pd_table.column('Photo', anchor='w')
pd_table.column('Class', anchor='w')
pd_table.column('Count', anchor='w')
pd_table.column('Confidence', anchor='w')

pd_table.heading('#0', text='', anchor='w')
pd_table.heading('Photo', text='Photo', anchor='w')
pd_table.heading('Class', text='Class', anchor='w')
pd_table.heading('Count', text='Count', anchor='w')
pd_table.heading('Confidence', text='Confidence', anchor='w')

pd_table.pack(fill='both', expand=True)


# Функция для обработки изображения и вызова нейросети
def process_image(model, image_path):
    # Вместо этой функции нужно подставить вызов вашей нейросети
    # и получение результатов (названия классов и количество объектов)

    classes = ['малый', 'кликун', 'шипун']
    labels, probs = inference_neural_network(model, images_paths=[image_path])
    return classes[labels[0].item()], 1, probs[0].item() * 100


# Функция для обновления таблицы
def update_table(model, image_path):
    class_name, count, confidence = process_image(model, image_path)
    image_name = os.path.basename(image_path)
    pd_table.insert("", "end", values=(image_name, class_name, count, confidence))

# Функция для сохранения таблицы в файл CSV


def save_table():
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    ROOT_DIR = os.getcwd()
    files = ROOT_DIR + '/' + 'labeled_images'
    os.makedirs(files, exist_ok=False)

    if file_path:
        data = []
        table = pd_table.get_children()
        for row in table:
            values = pd_table.item(row)['values']
            image_name = values[0]
            class_name = values[1]
            count = 1
            confidence = values[3]
            data.append([image_name, class_name, count, confidence])
            os.makedirs(files + '/' + confidence, exist_ok=True)
            shutil.copy(ROOT_DIR + '/' + image_name, files + '/' + confidence)

        df = pd.DataFrame(data, columns=['Photo', 'Class', 'Count', 'Confidence'])
        df.to_csv(file_path, index=False, delimeter=';')


# Функция для обработки папки с изображениями
def process_folder(folder_path):
    image_files = get_image_files(folder_path)
    total_images = len(image_files)
    processed_images = 0

    model = load_neural_network(num_classes=3, from_path=Path('pipeline/models/'), model_name='swan')

    for image_file in image_files:
        threading.Thread(target=update_table, args=(model, image_file)).start()
        processed_images += 1
        window.update_idletasks()

    progress_bar['value'] = 100


# Функция для получения списка файлов изображений в папке
def get_image_files(folder_path):
    image_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_files.append(os.path.join(folder_path, file))
    return image_files


# Создание кнопки для выбора папки с изображениями
def browse_folder():
    progress_bar['value'] = 0
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        threading.Thread(target=process_folder, args=(folder_selected,)).start()


def train_network():
    progress_bar['value'] = 0
    train_neural_network(classes_paths=[], save_path=Path('models/user'), model_name='user_model')
    progress_bar['value'] = 100


def make_prediction():
    progress_bar['value'] = 0
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        threading.Thread(target=process_folder, args=(folder_selected,)).start()


browse_button = tk.Button(window, text="Выбрать папку c датасетом", command=browse_folder)
browse_button.pack(pady=10)

train_button = tk.Button(window, text="Обучить нейросеть", command=train_network)
train_button.pack(pady=10)

# Создание кнопки для сохранения таблицы
save_button = tk.Button(window, text="Сделать предсказание", command=make_prediction)
save_button.pack(pady=10)

# Создание кнопки для сохранения таблицы
save_button = tk.Button(window, text="Сохранить таблицу", command=save_table)
save_button.pack(pady=10)

# Создание полоски загрузки
progress_bar = ttk.Progressbar(window, orient=tk.HORIZONTAL, length=300, mode='determinate')
progress_bar.pack(pady=10)

# Запуск основного цикла обработки событий
window.mainloop()
