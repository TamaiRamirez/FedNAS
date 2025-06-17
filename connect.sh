# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Error: No argument provided. Use your docker container name."
    echo "Usage: ./connect.sh <container_name>"
    exit 1
fi

docker exec -it $1 /bin/bash