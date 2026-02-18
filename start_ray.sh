#!/bin/bash
# start_ray.sh — run on each machine

HEAD_IP="192.168.0.114"
HOSTNAME=$(hostname)

case "$HOSTNAME" in
  robros-MS-7E59)
    ray start --head --port=6379
    ;;
  robros-5090)
    ray start --address=${HEAD_IP}:6379 \
      --resources='{"inference_pc": 1}'
    ;;
  *)
    echo "Unknown host: $HOSTNAME"
    exit 1
    ;;
esac