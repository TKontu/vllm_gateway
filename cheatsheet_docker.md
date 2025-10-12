docker-compose build --no-cache
docker save -o my-vllm-gateway.tar my-vllm-gateway:latest
docker load -i your-image.tar
