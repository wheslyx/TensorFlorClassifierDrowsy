#!/bin/sh
# git clone https://github.com/EscVM/OIDv4_ToolKit.git Clone Open Images Downloader OID
# Uninstall libraries from OID requirements.txt Do all installation in Python Virtual Environment venv or virtual-env
#python3 main.py downloader --classes 'Mobile phone' --type_csv train  do in virtual environment
# pip install --upgrade lxml uninstal and do it in virtual environment
#python3 oid_to_pascal_voc_xml.py these creates xml annotation from csv annotation
# I modifiy file in OID because "mobile phone" was counted as 1 letter but it is really 1 letter

#  AutoTrain.py

export PYTHONPATH=$PYTHONPATH:'pwd':'pwd'/slim # export object detection folder inside deteccion_pbjetos
# git clone https://github.com/puigalex/deteccion_objetos.git clone GitHub repository with Tensor Flow for automatic training algorithm "Object Classifier"
# cd deteccion_objetos/
# pip install panda
python3 xml_a_csv.py --inputs=img_test --output=test #generate xml files of testing Cellphone images
python3 xml_a_csv.py --inputs=img_entrenamiento --output=entrenamiento  #generate xml files of training Cellphone images
#  pip3 install pandas in python venv

python csv_a_tf.py --csv_input=CSV/test.csv --output_path=TFRecords/test.record --images=images #generate test cellphone TF file in TFRecords folder
python csv_a_tf.py --csv_input=CSV/entrenamiento.csv --output_path=TFRecords/entrenamiento.record --images=images #generate training cellphone TF file in TFRecords folder
python object_detection/train.py --logtostderr --train_dir=train --pipeline_config_path=modelo/faster_rcnn_resnet101_coco.config # train classifier algorithm for cellphone model using Tensor Flow. Checkpoints are stored in deteccion_objetos/train
python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path modelo/faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix train/model.ckpt-585 --output_directory modelo_congelado # Generates frozen model in modelo_congelado/frozen_inference_graph.pb to be used in evaluation of the algorithm
python object_detection/object_detection_runner.py # run algorithm evaluation from folder img_pruebas and store in output/img_pruebas
#  Created by Cesar Segura on 4/3/19.
#  echo site-packages >> /usr/local/lib/python3.7/site-packages/opencv3.pth
