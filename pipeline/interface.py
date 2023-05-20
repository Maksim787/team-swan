import torch
from pathlib import Path
from torch import nn
import PIL


def train_neural_network(classes_pathes: list[Path], save_path: Path) -> nn.Module:
    """
    Параметры:
    1. classes_pathes — пути к папкам с каждым классом
    В папке classes_pathes[i] лежат только картинки, соответствующие i-ому классу.
    Важно, что нумерация такая же, как и в classes_pathes, чтобы можно тот, кто звал функцию, имел mapping между номерами и названиями классов.
    2. save_path — путь, по которому нужно сохранить веса сети. Это не путь к папке, а путь к конечному файлу, который нужно создать.

    Что делает?
    1. Обучает нейронную сеть на classes_pathes. Нужная простая сеть, чтобы обучение было быстрым. Учим небольшое число эпох (1-5).
    2. Сохраняет веса сети в файл save_path.
    3. Возвращает обученную сеть

    Зачем нужно?
    1. Обучение нейросети на входных данных
    2. Сохранение ее весов
    """

    assert not save_path.exists()
    for path in classes_pathes:
        assert path.exists()
        path.is_dir()
    n_classes = len(classes_pathes)
    print(f'Number of classes: {n_classes}')
    print(f'Classes pathes: {classes_pathes}')
    # TODO: code to train neural network here
    raise NotImplementedError()

    return load_neural_network(save_path)  # можно оптимизировать: не загружать заново, а вернуть сеть, созданную в этой функции


def load_neural_network(from_path: Path) -> nn.Module:
    """
    Параметры:
    1. from_path — путь к файлу с нейросетью, который был создан в train_neural_network

    Что делает?
    1. Возвращает нейронную сеть, загруженную из from_path

    Зачем нужно?
    1. Загрузить модельку в память один раз, чтобы далее передавать в нее батчами списки картинок.
    """
    assert from_path.eixsts()
    assert from_path.is_file()

    # TODO: code to load neuarl network here
    raise NotImplementedError()


@torch.inference_mode()
def inference_neural_network(network: nn.Module, images: list[PIL.Image]) -> list[int]:
    """
    Параметры:
    1. network — сеть, загруженная через load_neural_network
    2. images — список изображений

    Что делает?
    1. Возвращает результаты классификации для всех изображений.
    Результат — номер класса в списке classes_pathes, на которых была обучена нейронная сеть
    """

    # TODO: code to make inference
    raise NotImplementedError()
