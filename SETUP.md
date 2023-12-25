## Running docker image

    docker build --tag 'deep_pavements' .

    docker run --gpus all -it 'deep_pavements'

include "--detach" to run in background and "--rm" to remove on exit
