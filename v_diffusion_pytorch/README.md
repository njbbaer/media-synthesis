# Media Synthesis

[crowsonkb/v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch)

## Setup

```bash
# Set environment variables
export SERVER_PORT={SERVER_PORT} SERVER_ADDRESS=root@{SERVER_ADDRESS}

# Connect to server
ssh -p $SERVER_PORT root@$SERVER_ADDRESS -L 8642:localhost:8642

# Disable tmux (disconnect and reconnect)
touch ~/.no_auto_tmux

# Download repository
apt install -y git
git clone https://github.com/njbbaer/media-synthesis.git
cd media-synthesis/v_diffusion_pytorch
```

## Operations

### Copy files to server

```bash
scp -r -P $SERVER_PORT ~/media-synthesis/guided_diffusion/scripts root@$SERVER_ADDRESS:/root
```

### Download images from server

```bash
scp -r -P $SERVER_PORT root@$SERVER_ADDRESS:/root/guided_diffusion/output ./
```
