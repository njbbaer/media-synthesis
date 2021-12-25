# Media Synthesis

[Nerdy Rodent VQGAN-CLIP](https://github.com/nerdyrodent/VQGAN-CLIP)

### Set environment variables

```bash
export SERVER_PORT=17847 SERVER_ADDRESS=root@ssh4.vast.ai
```

### Copy scripts to server

```bash
scp -r -P $SERVER_PORT ~/media-synthesis/VQGAN-CLIP/scripts $SERVER_ADDRESS:/root
```

### Download image from server

```bash
scp -r -P $SERVER_PORT $SERVER_ADDRESS:/root/VQGAN-CLIP/output ./
```
