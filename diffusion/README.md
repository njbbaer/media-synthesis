# Media Synthesis

[crowsonkb/v-diffusion-pytorch](https://github.com/crowsonkb/v-diffusion-pytorch)

## Setup

Set `SERVER_PORT` and `SERVER_ADDRESS` environment variables

Copy scripts to server

```bash
scp -r -q -P $SERVER_PORT ~/media-synthesis/diffusion/scripts/* root@$SERVER_ADDRESS:/root
```

SSH into server

```bash
ssh -p $SERVER_PORT root@$SERVER_ADDRESS -L 9341:localhost:9341
```

Run setup script

```bash
bash setup.sh
```

Start jupyter server

```bash
nohup jupyter lab --ip=127.0.0.1 --port=9341 --allow-root &
```

Print server URL and copy to VSCode

```bash
cat nohup.out
```

Close SSH connection

Open permenant SSH tunnel

```bash
autossh -M 0 -f -q -N -L 9341:localhost:9341 root@$SERVER_ADDRESS -p $SERVER_PORT
```
