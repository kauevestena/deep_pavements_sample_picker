## Running the docker image

first buid:

    docker build --tag 'deep_pavements' .

then run:

    docker run --name running_deep_pavements --mount type=bind,source=D:\segmentation_img_data,target=/home/data --gpus all -it 'deep_pavements' 

replace "D:\semantic_segmentation_data" with the desired path for mounting a volume were the outputs shall be generated. If you wanna more than one running container, you can remove "--name running_deep_pavements". 

include "--detach" to run in background and "--rm" to remove on exit

# Mapillary token:

create a file called "mapillary_token" in the project rootpath, containing your secret key.
For security reasons, the file is added to the .gitgnore
If it isn't created, the building process will fail. 