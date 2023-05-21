import torch
from pathlib import Path
from torch import nn
import PIL
import torch.nn.functional as F
from dataset_utils import get_dataloader
from model_utils import save_model, load_model
from train_utils_user import train, get_device
from model_configs import get_resnet18


def train_neural_network(classes_paths: list[Path], save_path: Path, model_name: str) -> nn.Module:
    """
    Параметры:
    1. classes_paths — пути к папкам с каждым классом
    В папке classes_paths[i] лежат только картинки, соответствующие i-ому классу.
    Важно, что нумерация такая же, как и в classes_paths, чтобы можно тот, кто звал функцию, имел mapping между номерами и названиями классов.
    2. save_path — путь, по которому нужно сохранить веса сети. Это не путь к папке, а путь к конечному файлу, который нужно создать.

    Что делает?
    1. Обучает нейронную сеть на classes_paths. Нужная простая сеть, чтобы обучение было быстрым. Учим небольшое число эпох (1-5).
    2. Сохраняет веса сети в файл save_path.
    3. Возвращает обученную сеть

    Зачем нужно?
    1. Обучение нейросети на входных данных
    2. Сохранение ее весов
    """

    assert not (save_path / f'{model_name}.pt').exists()
    for path in classes_paths:
        assert path.exists()
        path.is_dir()
    n_classes = len(classes_paths)
    print(f'Number of classes: {n_classes}')
    print(f'Classes pathes: {classes_paths}')

    # configure dataset
    data_loader = get_dataloader(classes_paths, is_test=False)

    # configure model
    model = get_resnet18(len(classes_paths)).to(get_device())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    # train model
    train(model, optimizer, criterion, data_loader)
    save_model(model, save_path, model_name)

    return load_neural_network(len(classes_paths), save_path, model_name)


def load_neural_network(num_classes: int, from_path: Path, model_name: str) -> nn.Module:
    """
    Параметры:
    1. from_path — путь к файлу с нейросетью, который был создан в train_neural_network

    Что делает?
    1. Возвращает нейронную сеть, загруженную из from_path

    Зачем нужно?
    1. Загрузить модельку в память один раз, чтобы далее передавать в нее батчами списки картинок.
    """
    assert (from_path / f'{model_name}.pt').is_file()

    model = get_resnet18(num_classes).to(get_device())
    return load_model(model, from_path, model_name)


@torch.inference_mode()
def inference_neural_network(model: nn.Module, images_paths: list[Path]) -> tuple:
    """
    Параметры:
    1. network — сеть, загруженная через load_neural_network
    2. images — список изображений

    Что делает?
    1. Возвращает результаты классификации для всех изображений.
    Результат — номер класса в списке classes_paths, на которых была обучена нейронная сеть
    """
    predicted_labels = []
    predicted_probs = []
    total_images = []
    device = get_device()

    # configure dataset
    data_loader = get_dataloader(images_paths, is_test=True)

    for images, labels in data_loader:
        images = images.to(device)
        output = model(images)
        predicted_labels.append(output.argmax(dim=1).cpu())
        total_images.extend(images)
    return torch.concat(predicted_labels), [torch.tensor(0.8)]
