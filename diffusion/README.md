# Media Synthesis

[crowsonkb/v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch)

## Setup

```bash
# Set environment variables
export SERVER_PORT=
export SERVER_ADDRESS=root@

# Copy scripts to server
scp -r -P $SERVER_PORT ~/media-synthesis/diffusion/scripts root@$SERVER_ADDRESS:/root

# Connect to server
ssh -p $SERVER_PORT root@$SERVER_ADDRESS -L 8642:localhost:8642

# Run setup script
bash setup.sh

# Start jupyter server
jupyter lab --ip=127.0.0.1 --port=8642 --allow-root
```

## Operations

### Download images from server

```bash
scp -r -q -P $SERVER_PORT root@$SERVER_ADDRESS:/root/diffusion/output ./
```
