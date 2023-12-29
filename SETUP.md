# Configuring:

Set the desired prompts at "configs/prompted_classes.csv", following the available model.
Set the desired territories at "configs/territories.csv", following the available model, territories must be geocodable in Nomitatim API, in case of more than one results, the one with biggest "relevance" will be chosen.  

In both cases watch for csv consistency (i recommend using CSVLint and RAinbowCSV).


# Mapillary token:

create a file called "mapillary_token" in the project rootpath, containing your secret key.
For security reasons, the file is added to the .gitgnore
If it isn't created, the building (check it below) process will fail. 


## Running the docker image

first buid:

    docker build --tag 'deep_pavements' .

then run:

    docker run --name running_deep_pavements --mount type=bind,source=D:\segmentation_img_data,target=/home/data --gpus all -it 'deep_pavements' 

replace "D:\semantic_segmentation_data" with the desired path for mounting a volume were the outputs shall be generated. If you wanna more than one running container, you can remove "--name running_deep_pavements". 

include "--detach" to run in background and "--rm" to remove on exit

# Running the sample colector:

Inside the container:

    python run.py

Outside shall be: 

    docker run running_deep_pavements python run.py
