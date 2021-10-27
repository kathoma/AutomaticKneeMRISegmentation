FROM tensorflow/tensorflow:2.0.0-gpu-py3
RUN apt-get update && apt-get -y install libglib2.0-dev libsm6 libxext6 libxrender1
RUN pip install imageio==2.6.1
RUN pip install Keras==2.3.1
RUN pip install Keras-Applications==1.0.8
RUN pip install Keras-Preprocessing==1.1.0
RUN pip install kiwisolver==1.1.0
RUN pip install matplotlib==3.1.1
RUN pip install nbconvert==5.6.1
RUN pip install nbformat==4.4.0
RUN pip install nibabel==3.1.0
RUN pip install numpy==1.17.4
RUN pip install pandas==0.25.3
RUN pip install Pillow==7.0.0
RUN pip install pydicom==1.3.0
RUN pip install scikit-image==0.15.0
RUN pip install scikit-learn==0.22.1
RUN pip install scipy==1.4.1

CMD ["python3", "-u", "/workspace/predict.py"]