# Media Synthesis

[crowsonkb/v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch)

## Setup

```bash
# Set environment variables
export SERVER_PORT=
export SERVER_ADDRESS=

# Copy scripts to server
scp -r -q -P $SERVER_PORT ~/media-synthesis/diffusion/scripts/* root@$SERVER_ADDRESS:/root

# Connect to server
ssh -p $SERVER_PORT root@$SERVER_ADDRESS -L 9341:localhost:9341

# Run setup script
bash setup.sh

# Start jupyter server
jupyter lab --ip=127.0.0.1 --port=9341 --allow-root
```
