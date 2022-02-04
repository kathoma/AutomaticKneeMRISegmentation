if [ ! -f model_weights/model_weights_quartileNormalization_echoAug.h5 ]; then
    mkdir -p model_weights
    mkdir -p output
    curl https://storage.googleapis.com/automatic_knee_mri_segmentation/model_weights_quartileNormalization_echoAug.h5 -o model_weights/model_weights_quartileNormalization_echoAug.h5  
fi

docker build -t kneeseg .

docker run \
    -v ${PWD}:/workspace \
    -v /home/ubuntu/bigger_data/input:/workspace/input \
    -v /home/ubuntu/bigger_data/output:/workspace/output \
    -v ${PWD}/model_weights:/workspace/model_weights \
    -t kneeseg 
    
# && printf "\n-- RESULTS (in output/prediction.csv) --\n"

#     python scripts/predict.py && printf "\n-- RESULTS (in output/prediction.csv) --\n"
#     -v ${PWD}/input:/workspace/input \
#     -v ${PWD}/output:/workspace/output \
#     -v ${PWD}/model_weights:/workspace/model_weights \
