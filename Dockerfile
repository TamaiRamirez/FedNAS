FROM tensorflow/tensorflow:2.15.0-gpu
RUN apt-get update && apt-get install -y 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install ipykernel -U --user --force-reinstall
RUN pip install -U ipywidgets
RUN pip install -U absl-py
RUN pip install -U tensorboard
RUN pip install -U scikit-learn
RUN pip install -U np_utils
RUN pip install -U matplotlib
RUN pip install -U pandas
RUN pip install -U opencv-python
RUN pip install -U flwr[simulation]==1.8.0
RUN pip install -U flwr-datasets[vision]
RUN pip install -U hydra-core
RUN pip install -U memory_profiler

#Librerias para la practica Master de IA
RUN pip install -U mss
RUN pip install -U pynput
RUN pip install -U ultralytics
