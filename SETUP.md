# Configuring:

Refer to configs/ABOUT.md

# Mapillary token:

create a file called "mapillary_token" in the project rootpath, containing your secret key.
For security reasons, the file is added to the .gitgnore
If it isn't created, the building (check it below) process will fail. 


## Running the docker image

first, buid:

    docker build --tag 'deep_pavements_sample_picker' .

then run:

    docker run --name running_deep_pavements_sp -v D:\segmentation_img_data:/workspace/data --gpus all -it 'deep_pavements_sample_picker' 

replace "D:\semantic_segmentation_data" with the desired path for mounting a volume where the outputs shall be generated. You can remove "--name running_deep_pavements" if you want multiple running containers. 

include "--detach" to run in the background and "--rm" to remove on exit

# Running the sample collector:

Inside the container:

    python run.py

Outside it shall be: 

    docker run running_deep_pavements python run.py
