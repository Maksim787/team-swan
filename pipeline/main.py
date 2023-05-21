from pathlib import Path

from interface import train_neural_network, inference_neural_network, load_neural_network

train_neural_network(
    [
        Path('C:\\Users\\AsyaRita\\Downloads\\data\\klikun\\images'),
        Path('C:\\Users\\AsyaRita\\Downloads\\data\\maliy\\images'),
        Path('C:\\Users\\AsyaRita\\Downloads\\data\\shipun\\images')
    ],
    Path('C:\\Users\\AsyaRita\\Downloads\\data'),
    'my_model')

print(
    inference_neural_network(
        load_neural_network(3, Path('C:\\Users\\AsyaRita\\Downloads\\data'), 'my_model'),
        [Path('C:\\Users\\AsyaRita\\Downloads\\data\\klikun\\images')]
    )
)
